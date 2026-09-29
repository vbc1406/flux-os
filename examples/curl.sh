#!/usr/bin/env sh
# flux-os serve   (in another terminal)
BASE=${BASE:-http://localhost:8000}

# Which model would this go to? (no model call)
curl -s $BASE/v1/route -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Prove there are infinitely many primes"}]}'
echo

# OpenAI format — routing details come back in x-flux-* headers
curl -si $BASE/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hi"}]}' | grep -iE '^x-flux|content'

# Anthropic format
curl -s $BASE/v1/messages -H 'content-type: application/json' \
  -d '{"model":"auto","max_tokens":100,"messages":[{"role":"user","content":"Say hi"}]}'
echo
