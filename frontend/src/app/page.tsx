"use client";

import { Show, SignInButton, UserButton, useAuth } from "@clerk/nextjs";
import { useState } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

const EXAMPLE_QUESTIONS = [
  { q: "What BMI is required for bariatric surgery coverage?", emoji: "⚖️" },
  { q: "Is a total knee replacement covered without physical therapy first?", emoji: "🦵" },
  { q: "What are the coverage criteria for oxygen and oxygen equipment?", emoji: "🫁" },
  { q: "Is heart bypass surgery covered?", emoji: "❤️" },
];

interface UnsupportedCitation {
  chunk_id?: string;
  claim?: string;
  reason?: string;
}

interface QueryResponse {
  answer: string;
  sub_questions: string[];
  cited_chunk_ids: string[];
  phantom_citations: string[];
  unsupported_citations: UnsupportedCitation[];
  viewer_role: string;
}

export default function Home() {
  const { getToken } = useAuth();
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function submitQuestion(q: string) {
    if (!q.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // Signed-in: attach a real, verifiable Clerk session token. Signed
      // out: no Authorization header at all -- the API scopes that to
      // standard access on its own (see api.py's ANONYMOUS_USER), it isn't
      // something the client claims.
      const token = await getToken();
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`${res.status}: ${detail}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-6 py-14">
      <header className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 via-fuchsia-500 to-cyan-400 text-xl shadow-lg shadow-violet-500/30">
            🩺
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight">
            <span className="bg-gradient-to-r from-violet-500 via-fuchsia-500 to-cyan-400 bg-clip-text text-transparent">
              CoverageCounsel
            </span>
          </h1>
        </div>
        <p className="text-sm leading-relaxed text-foreground/70">
          Ask a question about real CMS Medicare coverage policy. Answers are grounded in
          retrieved policy excerpts, cited inline, and checked for faithfulness before display.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submitQuestion(question);
        }}
        className="flex flex-col gap-4 rounded-2xl border border-border bg-card/70 p-5 shadow-xl shadow-violet-950/5 backdrop-blur-sm"
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What BMI is required for bariatric surgery coverage?"
          rows={3}
          className="w-full resize-none rounded-xl border border-border bg-background/60 p-4 text-sm outline-none transition-all placeholder:text-foreground/35 focus:border-violet-400 focus:ring-4 focus:ring-violet-400/15"
        />

        <div className="flex flex-wrap items-center gap-3">
          <Show when="signed-out">
            <SignInButton mode="modal">
              <button
                type="button"
                className="flex items-center gap-1.5 rounded-full border border-border bg-background/60 px-3 py-1.5 text-xs font-medium text-foreground/70 transition-all hover:border-violet-400/60 hover:text-foreground"
              >
                🔐 Sign in
              </button>
            </SignInButton>
          </Show>
          <Show when="signed-in">
            <div className="flex items-center gap-2 rounded-full border border-border bg-background/60 py-1 pr-3 pl-1 text-xs text-foreground/70">
              <UserButton />
              signed in
            </div>
          </Show>

          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="ml-auto flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-fuchsia-500/25 transition-all hover:scale-[1.03] hover:shadow-fuchsia-500/40 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
          >
            {loading ? (
              <>
                <span className="flex gap-0.5">
                  <span className="loading-dot h-1.5 w-1.5 rounded-full bg-white" style={{ animationDelay: "0ms" }} />
                  <span className="loading-dot h-1.5 w-1.5 rounded-full bg-white" style={{ animationDelay: "150ms" }} />
                  <span className="loading-dot h-1.5 w-1.5 rounded-full bg-white" style={{ animationDelay: "300ms" }} />
                </span>
                Thinking
              </>
            ) : (
              <>Ask →</>
            )}
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {EXAMPLE_QUESTIONS.map(({ q, emoji }) => (
            <button
              key={q}
              type="button"
              onClick={() => {
                setQuestion(q);
                submitQuestion(q);
              }}
              className="rounded-full border border-border bg-background/50 px-3 py-1.5 text-xs text-foreground/70 transition-all hover:-translate-y-0.5 hover:border-violet-400/60 hover:text-foreground hover:shadow-md hover:shadow-violet-500/10"
            >
              <span className="mr-1">{emoji}</span>
              {q}
            </button>
          ))}
        </div>
      </form>

      {error && (
        <div className="animate-fade-up rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-500">
          <span className="font-semibold">⚠ Error: </span>
          {error}
        </div>
      )}

      {result && (
        <div className="animate-fade-up flex flex-col gap-5 rounded-2xl border border-border bg-card/70 p-6 shadow-xl shadow-violet-950/5 backdrop-blur-sm">
          <div className="flex items-center gap-1.5 text-[11px] text-foreground/45">
            <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />
            Answered with <span className="font-medium text-foreground/60">{result.viewer_role}</span> access
          </div>

          {result.sub_questions.length > 1 && (
            <div className="flex flex-wrap items-center gap-2 text-xs text-foreground/60">
              <span className="rounded-full bg-cyan-500/15 px-2.5 py-1 font-medium text-cyan-500">
                Decomposed into {result.sub_questions.length}
              </span>
              {result.sub_questions.map((sq, i) => (
                <span key={i} className="rounded-full border border-border px-2.5 py-1">
                  {sq}
                </span>
              ))}
            </div>
          )}

          <div className="prose prose-sm max-w-none prose-neutral dark:prose-invert prose-headings:bg-gradient-to-r prose-headings:from-violet-500 prose-headings:to-fuchsia-500 prose-headings:bg-clip-text prose-headings:text-transparent prose-strong:text-foreground">
            <ReactMarkdown>{result.answer}</ReactMarkdown>
          </div>

          {result.cited_chunk_ids.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 border-t border-border pt-4 text-xs">
              <span className="mr-1 font-medium text-foreground/50">📎 Sources:</span>
              {result.cited_chunk_ids.map((id) => (
                <span
                  key={id}
                  className="rounded-md bg-violet-500/10 px-2 py-1 font-mono text-[11px] text-violet-500 dark:text-violet-300"
                >
                  {id}
                </span>
              ))}
            </div>
          )}

          {result.phantom_citations.length > 0 && (
            <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-3 text-xs text-amber-600 dark:text-amber-400">
              <span className="font-semibold">⚠ Phantom citations detected: </span>
              {result.phantom_citations.join(", ")} — cited but never actually retrieved.
            </div>
          )}

          {result.unsupported_citations.length > 0 && (
            <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-3 text-xs text-amber-600 dark:text-amber-400">
              <span className="font-semibold">
                ⚠ Faithfulness check flagged {result.unsupported_citations.length} citation(s):
              </span>
              <ul className="mt-1.5 list-disc space-y-0.5 pl-4">
                {result.unsupported_citations.map((c, i) => (
                  <li key={i}>
                    <span className="font-mono">[{c.chunk_id}]</span> {c.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <footer className="mt-auto flex items-center gap-2 pt-8 text-xs text-foreground/40">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        Real CMS National/Local Coverage Determinations · hybrid retrieval + reranking ·
        LangGraph decomposition · RBAC at the retrieval layer ·{" "}
        <a
          href="https://github.com/shr7q/CoverageCounsel"
          className="font-medium text-violet-500 underline decoration-violet-500/30 underline-offset-2 hover:text-violet-400"
        >
          Source
        </a>
      </footer>
    </div>
  );
}
