// lfm3-1.7b-mi50 serves LFM3-1.7B on an AMD MI50 from one self-contained static executable.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime/debug"
	"time"

	"lfm3-1.7b-mi50/internal/model"
	"lfm3-1.7b-mi50/internal/payload"
	"lfm3-1.7b-mi50/internal/server"
	"lfm3-1.7b-mi50/internal/tokenizer"
)

func main() {
	host := flag.String("host", "0.0.0.0", "listen address")
	port := flag.Int("port", 18000, "listen port")
	maxContext := flag.Int("max-model-len", 131072, "longest prompt + output in tokens")
	vram := flag.Float64("gpu-memory-utilization", 0.90, "fraction of VRAM this process may use")
	payloadPath := flag.String("payload", "", "read the model from this file instead of the executable (development)")
	kvTokens := flag.Int("kv-tokens", 0, "full-attention KV capacity in tokens (0: use all remaining VRAM budget)")
	noDraft := flag.Bool("no-draft", false, "serve greedy requests without DSpark speculation")
	exactAttention := flag.Bool("exact-attention", false, "prefill attention with fp16 softmax probabilities: about half the attention speed, the previous long-context error")
	flag.Parse()

	// The heap holds only small request state; collecting rarely keeps the token loop clean.
	debug.SetGCPercent(400)

	path := *payloadPath
	if path == "" {
		self, err := os.Executable()
		if err != nil {
			log.Fatal(err)
		}
		path = self
	}
	pay, err := payload.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	raw, err := pay.Data("tokenizer")
	if err != nil {
		log.Fatal(err)
	}
	tok, err := tokenizer.Load(raw)
	if err != nil {
		log.Fatal(err)
	}
	start := time.Now()
	engine, err := model.Open(model.Config{MaxContext: *maxContext, VRAMFraction: *vram, FullTokens: *kvTokens, NoDraft: *noDraft, ExactAttention: *exactAttention}, pay)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("model ready in %.1f s, KV capacity %d tokens", time.Since(start).Seconds(), engine.KVCapacityTokens())
	srv := server.New(engine, tok, *maxContext)
	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("serving %s on http://%s", server.ModelName, addr)
	log.Fatal((&http.Server{Addr: addr, Handler: srv.Handler()}).ListenAndServe())
}
