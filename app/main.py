"""政策合规问答 · 检索增强服务。

接口：
    GET  /             简易问答页面（用于演示录屏/截图）
    GET  /api/health   健康检查与索引统计
    GET  /api/docs     文档清单
    POST /api/search   只检索不生成，用于调试召回
    POST /api/chat     完整问答，返回答案 + 引用 + 召回明细
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import settings
from .embedder import build_embedder
from .index import HybridRetriever
from .llm import build_models
from .rag import RAGPipeline

STATE: dict[str, Any] = {}


def load_pipeline() -> RAGPipeline:
    retriever = HybridRetriever.load(
        settings.index_dir, settings.bm25_k1, settings.bm25_b, settings.rrf_k
    )
    embedder = build_embedder(settings)
    models = build_models(settings)
    return RAGPipeline(retriever, embedder, models.llm, models.reranker, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["pipeline"] = load_pipeline()
        STATE["error"] = ""
        print(f"索引加载完成：{len(STATE['pipeline'].retriever.chunks)} 个检索块")
    except Exception as exc:  # noqa: BLE001
        STATE["pipeline"] = None
        STATE["error"] = str(exc)
        print(f"索引加载失败：{exc}")
    yield
    STATE.clear()


app = FastAPI(title="工程政策合规问答", version="1.0.0", lifespan=lifespan)


def _pipe() -> RAGPipeline:
    pipe = STATE.get("pipeline")
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail=f"索引未就绪：{STATE.get('error', '')}。请先执行 python -m scripts.ingest",
        )
    return pipe


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_ids: list[str] | None = Field(default=None, description="限定在指定文档内检索")


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


@app.get("/api/health")
def health() -> dict[str, Any]:
    pipe = STATE.get("pipeline")
    if pipe is None:
        return {"status": "error", "detail": STATE.get("error", "")}
    chunks = pipe.retriever.chunks
    return {
        "status": "ok",
        "chunks": len(chunks),
        "documents": len({c.clause.doc_id for c in chunks}),
        "embedding": pipe.embedder.name,
        "rerank": bool(pipe.reranker),
        "llm": pipe.llm.model if pipe.llm else "未配置（抽取式降级）",
    }


@app.get("/api/docs")
def list_docs() -> dict[str, Any]:
    pipe = _pipe()
    path = Path(settings.index_dir) / "docs.json"
    if path.exists():
        return {"documents": json.loads(path.read_text(encoding="utf-8"))}
    seen: dict[str, Any] = {}
    for chunk in pipe.retriever.chunks:
        clause = chunk.clause
        seen.setdefault(
            clause.doc_id,
            {
                "doc_id": clause.doc_id,
                "title": clause.doc_title,
                "doc_no": clause.doc_no,
                "issuer": clause.issuer,
                "publish_date": clause.publish_date,
            },
        )
    return {"documents": list(seen.values())}


@app.post("/api/search")
def search(req: SearchRequest) -> dict[str, Any]:
    pipe = _pipe()
    hits = pipe.search(req.question, top_k=req.top_k)
    return {
        "question": req.question,
        "hits": [
            {
                "rank": h.rank,
                "score": round(h.score, 6),
                "source": h.source,
                "citation": h.clause.citation if h.clause else "",
                "is_repealed": bool(h.clause and h.clause.is_repealed),
                "text": pipe.chunks_by_id[h.chunk_id].body[:400],
            }
            for h in hits
        ],
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    pipe = _pipe()
    answer = pipe.answer(req.question, doc_filter=set(req.doc_ids) if req.doc_ids else None)
    return answer.to_dict()


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>工程政策合规问答</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#262b36; --fg:#e6e9ef; --muted:#8b94a6; --accent:#4f8cff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 80px; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 22px; }
  .box { background: var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }
  textarea { width:100%; min-height:88px; resize:vertical; background:#0d1014; color:var(--fg);
             border:1px solid var(--line); border-radius:8px; padding:12px; font-size:14px;
             font-family:inherit; outline:none; }
  textarea:focus { border-color: var(--accent); }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px;
           padding:9px 20px; font-size:14px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .hint { color:var(--muted); font-size:12px; }
  .ans { margin-top:18px; white-space:pre-wrap; line-height:1.75; font-size:14.5px; }
  .refused { color:#ffb454; }
  .cite { margin-top:14px; border-top:1px dashed var(--line); padding-top:12px; }
  .cite h3 { font-size:13px; color:var(--muted); margin:0 0 8px; font-weight:600; letter-spacing:.5px; }
  .cite li { font-size:13px; color:#b9c4d8; margin-bottom:4px; }
  .dead { color:#ff6b6b; }
  .meta { margin-top:10px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>工程政策合规问答</h1>
  <div class="sub">条款级检索 · 引用可校验 · 证据不足直接拒答</div>
  <div class="box">
    <textarea id="q" placeholder="例如：建筑劳务企业没按时报送实名制信息，会被怎么处理？"></textarea>
    <div class="row">
      <button id="go">查询</button>
      <span class="hint" id="hint">Enter 发送 · Shift+Enter 换行</span>
    </div>
  </div>
  <div id="out"></div>
</div>
<script>
const q = document.getElementById('q'), out = document.getElementById('out'), go = document.getElementById('go');
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
go.addEventListener('click', ask);
async function ask() {
  const question = q.value.trim();
  if (!question) return;
  go.disabled = true; out.innerHTML = '<div class="meta">检索中…</div>';
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question, top_k:5})
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || '请求失败');
    const cls = d.refused ? 'ans refused' : 'ans';
    let html = '<div class="box" style="margin-top:18px"><div class="' + cls + '">'
             + esc(d.answer) + '</div>';
    if (d.citations && d.citations.length) {
      html += '<div class="cite"><h3>引用条款</h3><ul>';
      for (const c of d.citations) {
        html += '<li' + (c.is_repealed ? ' class="dead"' : '') + '>'
              + esc(c.citation) + (c.is_repealed ? '（已废止）' : '') + '</li>';
      }
      html += '</ul></div>';
    }
    html += '<div class="meta">检索 ' + d.retrieved.length + ' 条 · 耗时 '
          + d.latency_ms + 'ms'
          + (d.debug && d.debug.mode ? ' · 模式 ' + esc(d.debug.mode) : '')
          + '</div></div>';
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = '<div class="meta">出错了：' + esc(err.message) + '</div>';
  } finally { go.disabled = false; }
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}
</script>
</body>
</html>"""
