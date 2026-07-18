# SSL Chatbot — Vercel Frontend

Static frontend that talks to the Flask backend deployed on Hugging Face Spaces.
This includes the chat UI plus the dashboard pages:

- `index.html`
- `dashboard.html`
- `dashboard-detail.html`

## Configure the backend URL

In [`index.html`](index.html), set:

```html
<script>
  window.API_BASE = "https://<your-user>-<your-space>.hf.space";
</script>
```

Leave it as `""` if you want the frontend to call the same origin as the page (useful for local testing where the backend is also serving the page).

## Deploy to Vercel

```bash
cd frontend
npx vercel            # first deploy (preview)
npx vercel --prod     # production
```

Or import this folder via the Vercel dashboard — it's a pure static site, no build step required. Vercel auto-serves `index.html` and the `static/` directory.

## Local preview

```bash
cd frontend
python3 -m http.server 5173
# open http://localhost:5173
```

Make sure the backend (Hugging Face Space) allows your local origin in `CORS_ORIGINS`, e.g. `CORS_ORIGINS=http://localhost:5173,https://your-vercel-app.vercel.app`.
