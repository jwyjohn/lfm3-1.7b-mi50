// bench replays token-id prompts against any OpenAI-compatible server and reports
// absolute single-stream numbers: TTFT, prefill tok/s and decode tok/s.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type caseLine struct {
	SessionID    string         `json:"session_id"`
	OutputLength int            `json:"output_length"`
	Payload      map[string]any `json:"payload"`
}

type result struct {
	Session string  `json:"session"`
	Input   int     `json:"input"`
	Output  int     `json:"output"`
	TTFT    float64 `json:"ttft_s"`
	Decode  float64 `json:"decode_tps"`
	Cached  int     `json:"cached"`
	Tokens  []int32 `json:"tokens"`
}

type chunk struct {
	Choices []struct {
		Text     string  `json:"text"`
		TokenIDs []int32 `json:"token_ids"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens  int `json:"prompt_tokens"`
		PromptDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

func run(url string, c caseLine, salt string, keep int) (*result, error) {
	c.Payload["cache_salt"] = salt
	c.Payload["stream"] = true
	c.Payload["stream_options"] = map[string]any{"include_usage": true}
	c.Payload["return_token_ids"] = true
	delete(c.Payload, "min_tokens")
	body, err := json.Marshal(c.Payload)
	if err != nil {
		return nil, err
	}
	start := time.Now()
	resp, err := http.Post(url+"/v1/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		var buf bytes.Buffer
		buf.ReadFrom(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %.300s", resp.StatusCode, buf.String())
	}
	r := &result{Session: c.SessionID}
	prompt, _ := c.Payload["prompt"].([]any)
	r.Input = len(prompt)
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	var first, last time.Time
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var ch chunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			return nil, fmt.Errorf("bad chunk %.120s: %w", data, err)
		}
		if ch.Usage != nil && ch.Usage.PromptDetails != nil {
			r.Cached = ch.Usage.PromptDetails.CachedTokens
		}
		for _, cc := range ch.Choices {
			n := len(cc.TokenIDs)
			if n == 0 {
				continue
			}
			now := time.Now()
			if r.Output == 0 {
				first = now
				r.TTFT = now.Sub(start).Seconds()
			}
			r.Output += n
			last = now
			if len(r.Tokens) < keep {
				r.Tokens = append(r.Tokens, cc.TokenIDs...)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if r.Output > 1 && last.After(first) {
		r.Decode = float64(r.Output-1) / last.Sub(first).Seconds()
	}
	return r, nil
}

func median(v []float64) float64 {
	sort.Float64s(v)
	return v[len(v)/2]
}

func main() {
	url := flag.String("url", "http://127.0.0.1:18000", "server base URL")
	dir := flag.String("cases", "", "directory of <case>.jsonl files")
	names := flag.String("run", "single-t1024-o1024,single-t8192-o1024,single-t32768-o1024,single-t65536-o1024,single-t128000-o1024", "comma-separated cases")
	out := flag.String("out", "", "JSON results file")
	keep := flag.Int("keep", 256, "output tokens to record per request")
	warm := flag.Bool("warm", false, "share one cache salt per session id prefix instead of a fresh salt per request")
	flag.Parse()
	var all []*result
	stamp := time.Now().UnixNano()
	fmt.Printf("%-28s %8s %9s %11s %11s\n", "case", "input", "TTFT s", "prefill t/s", "decode t/s")
	for _, name := range strings.Split(*names, ",") {
		f, err := os.Open(filepath.Join(*dir, name+".jsonl"))
		if err != nil {
			log.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<28)
		var ttft, pre, dec []float64
		input := 0
		i := 0
		for sc.Scan() {
			var c caseLine
			if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
				log.Fatal(err)
			}
			// Cold: a salt unique to this request. Warm: one salt per trace, so
			// consecutive turns of a session share their prefix.
			salt := fmt.Sprintf("bench-%d-%s-%d", stamp, name, i)
			if *warm {
				salt = fmt.Sprintf("bench-%d-%s", stamp, c.SessionID)
			}
			i++
			r, err := run(*url, c, salt, *keep)
			if err != nil {
				log.Fatalf("%s: %v", c.SessionID, err)
			}
			all = append(all, r)
			ttft = append(ttft, r.TTFT)
			pre = append(pre, float64(r.Input)/r.TTFT)
			dec = append(dec, r.Decode)
			input = r.Input
			fmt.Printf("  %-26s %8d %9.3f %11.0f %11.1f  out=%d cached=%d\n", c.SessionID, r.Input, r.TTFT, float64(r.Input)/r.TTFT, r.Decode, r.Output, r.Cached)
		}
		f.Close()
		fmt.Printf("%-28s %8d %9.3f %11.0f %11.1f\n", name, input, median(ttft), median(pre), median(dec))
	}
	if *out != "" {
		raw, err := json.Marshal(all)
		if err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*out, raw, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
