/* 云策知识库 · 前端交互 */
(function () {
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const chat = document.getElementById('chat');
  const statusText = document.getElementById('status-text');
  const statusDot = document.querySelector('.status-dot');
  const exampleChips = document.querySelectorAll('.example-chip');
  const historyList = document.getElementById('history-list');
  const historyClear = document.getElementById('history-clear');
  const historySearch = document.getElementById('history-search');
  const historyPager = document.getElementById('history-pager');
  const newChatBtn = document.getElementById('new-chat');

  let sending = false;
  let chatHistory = [];      // 当前对话 [{role, content}]
  let historyPage = 1;
  let historyKeyword = '';

  /* ---------- 示例问题 ---------- */
  exampleChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      input.value = chip.textContent.trim();
      autoResize();
      send();
    });
  });

  /* ---------- 输入 ---------- */
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  input.addEventListener('input', autoResize);
  sendBtn.addEventListener('click', send);

  function autoResize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  }

  /* ---------- 状态灯 ---------- */
  function setStatus(state) {
    if (state === 'thinking') {
      statusText.textContent = 'Agent 执行中…';
      statusDot.style.background = 'var(--amber)';
      statusDot.style.animation = 'none';
      statusDot.style.boxShadow = '0 0 9px var(--amber)';
    } else {
      statusText.textContent = '系统就绪';
      statusDot.style.background = 'var(--green)';
      statusDot.style.animation = 'breathe 2.6s ease-in-out infinite';
      statusDot.style.boxShadow = '';
    }
  }

  /* ---------- 消息 ---------- */
  function addUserMsg(text) {
    const wrap = document.createElement('div');
    wrap.className = 'msg msg-user';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollToBottom();
  }

  function addAiMsg() {
    const wrap = document.createElement('div');
    wrap.className = 'msg msg-ai';
    const bubble = document.createElement('div');
    bubble.className = 'bubble loading';
    bubble.textContent = '正在思考…';
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollToBottom();
    return {
      setStreaming(text) {
        bubble.classList.remove('loading');
        bubble.innerHTML = renderMarkdown(text);
        scrollToBottom();
      },
      setAnswer(md) {
        bubble.classList.remove('loading');
        bubble.innerHTML = renderMarkdown(md);
        scrollToBottom();
      },
      setError(msg) {
        bubble.classList.remove('loading');
        bubble.innerHTML = '<span style="color:var(--red)">' + escapeHtml(msg) + '</span>';
        scrollToBottom();
      },
    };
  }

  /* ---------- 发送 ---------- */
  async function send() {
    const q = input.value.trim();
    if (!q || sending) return;
    sending = true;
    input.value = '';
    autoResize();

    addUserMsg(q);
    setStatus('thinking');
    const aiMsg = addAiMsg();

    const history = chatHistory.slice(); // 发送之前的历史（不含当前问题）
    try {
      const endpoint = '/api/agent/stream';
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, history: history }),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);

      // 流式读取 SSE，逐 token 渲染
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullAnswer = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();
        for (const ev of events) {
          const line = ev.trim();
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token) {
              fullAnswer += data.token;
              aiMsg.setStreaming(fullAnswer);
            } else if (data.error) {
              aiMsg.setError(data.error);
            }
          } catch (e) {
            /* 忽略解析失败的行 */
          }
        }
      }

      if (!fullAnswer) fullAnswer = '（无回答）';
      aiMsg.setAnswer(fullAnswer);
      chatHistory.push({ role: 'user', content: q });
      chatHistory.push({ role: 'assistant', content: fullAnswer });
    } catch (err) {
      aiMsg.setError('请求失败：' + err.message + '。请确认后端已启动（python server.py）');
    } finally {
      sending = false;
      setStatus('ready');
      loadHistory();
      input.focus();
    }
  }

  function scrollToBottom() {
    chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
  }

  /* ---------- 简单 Markdown 渲染 ---------- */
  function renderMarkdown(md) {
    if (!md) return '';
    const lines = md.split('\n');
    let html = '';
    let listOpen = false;

    const inline = (s) =>
      s
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.+?)`/g, '<code>$1</code>');

    for (const line of lines) {
      const t = line.trim();
      const h = t.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        if (listOpen) { html += '</ul>'; listOpen = false; }
        const tag = 'h' + (h[1].length + 1);
        html += '<' + tag + '>' + inline(h[2]) + '</' + tag + '>';
        continue;
      }
      const li = t.match(/^[-*]\s+(.*)$/);
      if (li) {
        if (!listOpen) { html += '<ul>'; listOpen = true; }
        html += '<li>' + inline(li[1]) + '</li>';
        continue;
      }
      if (!t) {
        if (listOpen) { html += '</ul>'; listOpen = false; }
        continue;
      }
      if (listOpen) { html += '</ul>'; listOpen = false; }
      html += '<p>' + inline(t) + '</p>';
    }
    if (listOpen) html += '</ul>';
    return html;
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------- 历史记录 ---------- */
  let cachedRecords = [];

  async function loadHistory() {
    try {
      const params = new URLSearchParams({
        keyword: historyKeyword,
        page: String(historyPage),
        page_size: '10',
      });
      const res = await fetch('/api/history?' + params.toString());
      const data = await res.json();
      cachedRecords = data.records || [];
      renderHistory(cachedRecords);
      renderPager(data);
    } catch (e) {
      /* 静默失败 */
    }
  }

  function renderPager(data) {
    const totalPages = Math.max(1, Math.ceil((data.total || 0) / (data.page_size || 10)));
    historyPager.innerHTML =
      '<button class="pager-btn" data-page="' + (data.page - 1) + '"' + (data.page <= 1 ? ' disabled' : '') + '>‹</button>' +
      '<span class="pager-info">' + data.page + ' / ' + totalPages + '</span>' +
      '<button class="pager-btn" data-page="' + (data.page + 1) + '"' + (!data.has_more ? ' disabled' : '') + '>›</button>';
    historyPager.querySelectorAll('.pager-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        if (btn.disabled) return;
        historyPage = parseInt(btn.dataset.page, 10);
        loadHistory();
      });
    });
  }

  function renderHistory(records) {
    if (!records.length) {
      historyList.innerHTML = '<div class="history-empty">暂无历史记录</div>';
      return;
    }
    historyList.innerHTML = records
      .map((r) => {
        const modeCls = r.mode === 'agent' ? 'agent' : 'ask';
        const modeLabel = r.mode === 'agent' ? '多Agent' : '单链';
        return (
          '<div class="history-item" data-id="' + r.id + '">' +
          '<div class="history-q">' + escapeHtml(r.question) + '</div>' +
          '<div class="history-meta">' +
          '<span class="history-tag ' + modeCls + '">' + modeLabel + '</span>' +
          '<span>' + formatTime(r.created_at) + '</span>' +
          '</div>' +
          '</div>'
        );
      })
      .join('');

    historyList.querySelectorAll('.history-item').forEach((item) => {
      item.addEventListener('click', () => {
        const id = parseInt(item.dataset.id, 10);
        const record = cachedRecords.find((r) => r.id === id);
        if (record) showHistoryDetail(record);
        historyList.querySelectorAll('.history-item').forEach((i) => i.classList.remove('active'));
        item.classList.add('active');
      });
    });
  }

  function showHistoryDetail(record) {
    chat.innerHTML = '';
    addUserMsg(record.question);
    addAiMsg().setAnswer(record.answer);
    scrollToBottom();
  }

  function formatTime(iso) {
    const d = new Date(iso.replace(' ', 'T'));
    if (isNaN(d.getTime())) return iso;
    const now = new Date();
    const hm = d.toTimeString().slice(0, 5);
    if (d.toDateString() === now.toDateString()) return hm;
    return (d.getMonth() + 1) + '/' + d.getDate() + ' ' + hm;
  }

  historyClear.addEventListener('click', async () => {
    if (!confirm('确定清空全部历史记录？')) return;
    try {
      await fetch('/api/history', { method: 'DELETE' });
      chat.innerHTML = '';
      loadHistory();
    } catch (e) {
      /* 静默失败 */
    }
  });

  // 历史搜索（防抖）
  let searchTimer;
  historySearch.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      historyKeyword = historySearch.value.trim();
      historyPage = 1;
      loadHistory();
    }, 400);
  });

  // 新对话
  newChatBtn.addEventListener('click', () => {
    chatHistory = [];
    chat.innerHTML = '';
    input.focus();
  });

  // 页面加载时加载历史
  loadHistory();
})();
