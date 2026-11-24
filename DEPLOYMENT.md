# Deploy to Render

1. Push this project to GitHub.
2. In Render, choose **New > Blueprint** and select the repository.
3. Render will read `render.yaml`.
4. Set `GROQ_API_KEY` to a valid Groq API key when Render asks for it.
5. Deploy the web service and open the generated HTTPS URL.

For microphone access, the app must be opened through HTTPS or localhost.

## Vercel

Vercel's serverless functions and static frontend model are not suitable for this
Streamlit app. It needs a persistent Python process with WebSockets. Use Render
for this project.
