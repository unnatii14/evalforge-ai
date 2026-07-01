# Deploying EvalForge free on Hugging Face Spaces (with Groq)

This deploys the EvalForge dashboard as a live, public web app for **$0**, using
[Groq](https://groq.com)'s free LLM API in place of local Ollama. No GPU, no
credit card, no server to manage.

Total time: ~15 minutes.

---

## 1. Get a free Groq API key

1. Go to <https://console.groq.com> and sign in (Google/GitHub login works).
2. Open **API Keys** → **Create API Key**. Name it `evalforge`.
3. Copy the key (starts with `gsk_...`). You'll paste it into Hugging Face in step 4.
   Keep it secret — never commit it to Git.

## 2. Push this project to GitHub (if it isn't already)

```powershell
git add .
git commit -m "Add Groq backend + Hugging Face Spaces deployment"
git push
```

## 3. Create the Hugging Face Space

1. Sign up / log in at <https://huggingface.co>.
2. Click your avatar → **New Space**.
3. Fill in:
   - **Owner**: your username
   - **Space name**: `evalforge`
   - **License**: MIT
   - **Space SDK**: choose **Docker** → **Blank**
   - **Hardware**: **CPU basic (free)**
   - **Visibility**: Public
4. Click **Create Space**.

## 4. Add your Groq key as a secret

In the new Space: **Settings** → **Variables and secrets** → **New secret**.

- **Name**: `GROQ_API_KEY`
- **Value**: the `gsk_...` key from step 1

Save. (The image already sets `EVAL_BACKEND=groq`, so you only need this one secret.)

## 5. Upload the code to the Space

The Space is its own Git repo. The simplest path:

```powershell
# from the project folder
git remote add space https://huggingface.co/spaces/<your-username>/evalforge
git push space main
```

If prompted for a password, use a Hugging Face **access token** (Settings →
Access Tokens → New token, role: write) as the password.

> Alternatively, use the Space's **Files → Upload files** button in the browser
> and drag in the whole project folder. Git push is cleaner.

## 6. Watch it build

The Space will show **Building** (installing dependencies takes a few minutes the
first time — it downloads the ML libraries and a small embedding model). When it
flips to **Running**, your app is live at:

```
https://<your-username>-evalforge.hf.space
```

The dashboard opens pre-seeded with two demo runs so it looks populated
immediately.

## 7. Run a real evaluation (the live demo)

1. In the app sidebar, open **Evaluation Runs**.
2. Set **LLM Provider** = `groq` (model auto-fills to `llama-3.3-70b-versatile`).
3. Pick the `halueval_rag_test` dataset, tick **RAGAS** and/or **DeepEval**.
4. Click **Run Active Evaluation**. It calls Groq, scores with RAGAS/DeepEval,
   and stores the run — refresh **Overview** to see it charted.

---

## Notes & troubleshooting

- **Free storage is ephemeral.** Runs you create may reset when the Space
  restarts; the demo seed re-populates automatically on boot. That's fine for a
  showcase. For permanence, attach HF persistent storage or point `SQLITE_PATH`
  at it.
- **Groq free-tier rate limits** apply. Keep `max_samples` small (5–10) for demos.
- **Model names change.** If a run errors with an unknown model, check the current
  list at <https://console.groq.com/docs/models> and update the model field (or the
  `GROQ_MODEL` variable in Space settings).
- **Everything still runs locally** with Ollama via `docker compose up` or the
  `uvicorn` + `streamlit` commands in the main README — the Groq backend is only
  activated by `EVAL_BACKEND=groq`.
