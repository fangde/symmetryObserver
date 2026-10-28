import os
import re
import html as html_lib
from markdown_it import MarkdownIt

def build_paper_html():
    source_path = "TENDON_PAPER_ZH.md"
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Cannot find {source_path}")
        
    with open(source_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 1. Extract and protect math equations
    math_blocks = []
    
    def repl_disp(m):
        idx = len(math_blocks)
        math_blocks.append((True, m.group(1)))
        return f"<!--MATH_DISP_{idx}-->"

    def repl_inline(m):
        idx = len(math_blocks)
        math_blocks.append((False, m.group(1)))
        return f"<!--MATH_INLINE_{idx}-->"

    # Match display math $$ ... $$ and inline math $ ... $
    protected_text = re.sub(r"\$\$(.*?)\$\$", repl_disp, md_text, flags=re.DOTALL)
    # Match inline math (prevent matching empty or currency)
    protected_text = re.sub(r"\$([^\$\n]+?)\$", repl_inline, protected_text)

    # 2. Render markdown to HTML with markdown_it
    md = MarkdownIt("commonmark", {"breaks": True, "html": True}).enable("table")
    content_html = md.render(protected_text)

    # 3. Restore math blocks
    for idx, (is_disp, content) in enumerate(math_blocks):
        clean_content = content.strip()
        if is_disp:
            content_html = content_html.replace(
                f"<!--MATH_DISP_{idx}-->", 
                f'<div class="math-block" data-math="{html_lib.escape(clean_content)}">$${clean_content}$$</div>'
            )
        else:
            content_html = content_html.replace(
                f"<!--MATH_INLINE_{idx}-->", 
                f'<span class="math-inline" data-math="{html_lib.escape(clean_content)}">${clean_content}$</span>'
            )

    # 4. Convert mermaid code blocks into <pre class="mermaid">
    def repl_mermaid(m):
        code = html_lib.unescape(m.group(1)).strip()
        return f'<div class="mermaid-container my-6 p-4 rounded-xl border border-border bg-card/60 flex justify-center overflow-x-auto"><pre class="mermaid">{code}</pre></div>'

    content_html = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>', repl_mermaid, content_html, flags=re.DOTALL)

    # 5. Add IDs to headings for Table of Contents
    toc_items = []
    def repl_heading(m):
        tag = m.group(1) # h1, h2, h3
        inner = m.group(2)
        # Extract clean text for ID
        text_plain = re.sub(r'<[^>]+>', '', inner)
        clean_id = re.sub(r'[^\w\u4e00-\u9fa5]+', '-', text_plain).strip('-').lower()
        if not clean_id:
            clean_id = f"heading-{len(toc_items)}"
        level = int(tag[1])
        if level in [2, 3]:
            toc_items.append((level, clean_id, text_plain))
        return f'<{tag} id="{clean_id}" class="scroll-mt-24 group flex items-center justify-between">{inner}<a href="#{clean_id}" class="opacity-0 group-hover:opacity-100 text-primary ml-2 text-sm no-underline transition-opacity">#</a></{tag}>'

    content_html = re.sub(r'<(h[1-4])>(.*?)</\1>', repl_heading, content_html, flags=re.DOTALL)

    # Build TOC HTML
    toc_html_list = []
    for level, clean_id, title in toc_items:
        indent = "pl-2 font-semibold text-foreground/90 mt-2" if level == 2 else "pl-6 text-muted-foreground text-xs hover:text-primary transition-colors"
        toc_html_list.append(f'<li class="{indent}"><a href="#{clean_id}" class="block py-1 truncate hover:underline">{title}</a></li>')
    toc_html = "\n".join(toc_html_list)

    # 6. Wrap tables in responsive div
    content_html = re.sub(
        r'<table>(.*?)</table>',
        r'<div class="table-container my-6 overflow-x-auto rounded-xl border border-border bg-card/40 shadow-sm"><table class="w-full text-sm text-left">\1</table></div>',
        content_html,
        flags=re.DOTALL
    )

    # Complete HTML template
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>基于受控对称性与级联正交微分同胚对齐的腱绳驱动关节高精度控制 | 论文全文</title>
  <meta name="description" content="通过微分流形对齐让复杂物理系统实现与简单系统相似的控制性能 - 完整学术长文">
  
  <!-- Tailwind CSS -->
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  
  <!-- KaTeX for LaTeX Math -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" 
    onload="renderMathInElement(document.body, {{
      delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}}
      ],
      throwOnError: false
    }});"></script>

  <!-- Mermaid.js -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'dark',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }});
  </script>

  <style>
    :root {{
      --background: #090d16;
      --foreground: #f1f5f9;
      --card: #131b2e;
      --card-muted: #0d1424;
      --border: #23314e;
      --muted-foreground: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #10b981;
    }}
    html.light {{
      --background: #ffffff;
      --foreground: #0f172a;
      --card: #f8fafc;
      --card-muted: #f1f5f9;
      --border: #e2e8f0;
      --muted-foreground: #64748b;
      --primary: #0284c7;
      --primary-hover: #0369a1;
      --accent: #059669;
    }}
    body {{
      background-color: var(--background);
      color: var(--foreground);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      line-height: 1.8;
    }}
    
    /* Academic Typography */
    .paper-content h1 {{
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      margin-top: 2.5rem;
      margin-bottom: 1.5rem;
      line-height: 1.3;
      border-bottom: 2px solid var(--border);
      padding-bottom: 0.8rem;
    }}
    .paper-content h2 {{
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-top: 2.75rem;
      margin-bottom: 1.25rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
      color: var(--foreground);
    }}
    .paper-content h3 {{
      font-size: 1.25rem;
      font-weight: 600;
      margin-top: 2rem;
      margin-bottom: 0.85rem;
      color: var(--primary);
    }}
    .paper-content h4 {{
      font-size: 1.05rem;
      font-weight: 600;
      margin-top: 1.5rem;
      margin-bottom: 0.5rem;
    }}
    .paper-content p {{
      margin-top: 1rem;
      margin-bottom: 1rem;
      text-align: justify;
      color: inherit;
    }}
    .paper-content ul, .paper-content ol {{
      margin-top: 0.75rem;
      margin-bottom: 1rem;
      padding-left: 1.75rem;
    }}
    .paper-content ul {{ list-style-type: disc; }}
    .paper-content ol {{ list-style-type: decimal; }}
    .paper-content li {{ margin-top: 0.35rem; margin-bottom: 0.35rem; }}
    .paper-content blockquote {{
      border-left: 4px solid var(--primary);
      background-color: var(--card-muted);
      padding: 1rem 1.5rem;
      border-radius: 0 0.75rem 0.75rem 0;
      margin: 1.5rem 0;
      font-style: normal;
      color: var(--foreground);
    }}
    .paper-content code:not(pre code) {{
      background-color: var(--card-muted);
      border: 1px solid var(--border);
      padding: 0.2rem 0.4rem;
      border-radius: 0.375rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.875em;
      color: var(--primary);
    }}
    .paper-content pre {{
      background-color: var(--card-muted);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      overflow-x: auto;
      margin: 1.5rem 0;
      font-size: 0.875rem;
      line-height: 1.6;
    }}
    .paper-content hr {{
      border: 0;
      height: 1px;
      background: var(--border);
      margin: 3rem 0;
    }}
    .paper-content img {{
      max-width: 100%;
      height: auto;
      border-radius: 0.75rem;
      border: 1px solid var(--border);
      margin: 1.5rem auto;
      display: block;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}
    .paper-content em {{
      display: block;
      text-align: center;
      font-size: 0.875rem;
      color: var(--muted-foreground);
      margin-top: -0.75rem;
      margin-bottom: 1.5rem;
    }}
    
    /* Tables */
    .table-container table {{
      border-collapse: collapse;
      width: 100%;
    }}
    .table-container th {{
      background-color: var(--card-muted);
      border-bottom: 2px solid var(--border);
      padding: 0.75rem 1rem;
      font-weight: 600;
      color: var(--foreground);
    }}
    .table-container td {{
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
    }}
    .table-container tr:hover td {{
      background-color: var(--card);
    }}
    
    /* Math Display */
    .math-block {{
      overflow-x: auto;
      padding: 1rem 0;
      margin: 1.25rem 0;
      text-align: center;
    }}
    
    /* Scroll progress bar */
    #progress-bar {{
      position: fixed;
      top: 0;
      left: 0;
      height: 3px;
      background: linear-gradient(to right, #38bdf8, #10b981);
      width: 0%;
      z-index: 100;
      transition: width 0.1s ease-out;
    }}
  </style>
</head>
<body class="antialiased min-h-screen">
  <div id="progress-bar"></div>

  <!-- Global Top Navigation Bar -->
  <nav class="sticky top-0 z-40 w-full border-b border-[var(--border)] bg-[var(--background)]/90 backdrop-blur-md">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <span class="px-2.5 py-1 text-xs font-bold rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
          PAPER FULL-TEXT
        </span>
        <a href="#" class="font-bold text-sm sm:text-base truncate max-w-xs sm:max-w-md hover:text-primary transition-colors">
          受控对称性与级联正交微分同胚对齐控制
        </a>
      </div>
      
      <div class="flex items-center gap-2 sm:gap-3 text-xs sm:text-sm">
        <a href="index.html" class="px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/20 font-medium transition-colors flex items-center gap-1.5">
          <span>📊</span> 实验测试报告
        </a>
        <a href="https://github.com/fangde/symmetryObserver" target="_blank" class="px-3 py-1.5 rounded-lg bg-[var(--card)] hover:bg-[var(--card-hover)] border border-[var(--border)] font-medium transition-colors flex items-center gap-1.5">
          <span>💻</span> GitHub
        </a>
        <button id="theme-toggle" class="p-2 rounded-lg bg-[var(--card)] hover:bg-[var(--card-hover)] border border-[var(--border)] text-muted-foreground hover:text-foreground transition-colors" title="切换浅色/深色主题">
          <span id="theme-icon">🌙</span>
        </button>
      </div>
    </div>
  </nav>

  <!-- Main Container -->
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12 flex flex-col lg:flex-row gap-10">
    
    <!-- Left Sticky Sidebar: Table of Contents -->
    <aside class="lg:w-80 shrink-0">
      <div class="sticky top-24 max-h-[calc(100vh-8rem)] overflow-y-auto pr-2 space-y-4 rounded-xl border border-[var(--border)] bg-[var(--card)]/50 p-5">
        <div class="flex items-center justify-between pb-3 border-b border-[var(--border)]">
          <span class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)] flex items-center gap-1.5">
            <span>📑</span> 目录导航 (Contents)
          </span>
          <span class="text-[10px] px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 font-mono">TOC</span>
        </div>
        <ul class="text-xs space-y-1" id="toc-list">
          {toc_html}
        </ul>
        <div class="pt-4 border-t border-[var(--border)] space-y-2">
          <div class="text-[11px] text-[var(--muted-foreground)]">
            作者：刘芳德 (Fangde Liu) & Antigravity 机器人动力学与控制研究组
          </div>
          <div class="text-[11px] text-[var(--muted-foreground)]">
            版本：2026年10月 正式版 (v1.2)
          </div>
        </div>
      </div>
    </aside>

    <!-- Right Main Column: Paper Article Content -->
    <main class="flex-1 min-w-0 max-w-4xl">
      <article class="paper-content bg-[var(--card)]/30 border border-[var(--border)] rounded-2xl p-6 sm:p-10 lg:p-12 shadow-xl">
        {content_html}
      </article>

      <!-- Footer -->
      <footer class="mt-12 pt-8 border-t border-[var(--border)] text-center text-xs text-[var(--muted-foreground)] space-y-2">
        <p>© 2026 Controlled Symmetry Research Project. Released under Open Robotics Research License.</p>
        <p>
          <a href="index.html" class="text-primary hover:underline">返回交互式测试报告 (Interactive Benchmark)</a> · 
          <a href="https://github.com/fangde/symmetryObserver" class="text-primary hover:underline">GitHub 代码仓库</a> · 
          <a href="TENDON_PAPER_ZH.md" class="text-primary hover:underline">下载 Markdown 原文</a>
        </p>
      </footer>
    </main>

  </div>

  <script>
    // 1. Reading Progress Bar
    window.addEventListener('scroll', () => {{
      const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      document.getElementById('progress-bar').style.width = scrolled + '%';
    }});

    // 2. Light / Dark Theme Toggle
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const htmlElem = document.documentElement;

    themeToggleBtn.addEventListener('click', () => {{
      if (htmlElem.classList.contains('dark')) {{
        htmlElem.classList.remove('dark');
        htmlElem.classList.add('light');
        themeIcon.textContent = '☀️';
        localStorage.setItem('theme', 'light');
      }} else {{
        htmlElem.classList.remove('light');
        htmlElem.classList.add('dark');
        themeIcon.textContent = '🌙';
        localStorage.setItem('theme', 'dark');
      }}
    }});

    if (localStorage.getItem('theme') === 'light') {{
      htmlElem.classList.remove('dark');
      htmlElem.classList.add('light');
      themeIcon.textContent = '☀️';
    }}

    // 3. Highlight Active TOC Item on Scroll
    const observer = new IntersectionObserver((entries) => {{
      entries.forEach(entry => {{
        if (entry.isIntersecting) {{
          const id = entry.target.getAttribute('id');
          document.querySelectorAll('#toc-list a').forEach(a => {{
            if (a.getAttribute('href') === '#' + id) {{
              a.classList.add('text-primary', 'font-bold');
            }} else {{
              a.classList.remove('text-primary', 'font-bold');
            }}
          }});
        }}
      }});
    }}, {{ rootMargin: '0px 0px -80% 0px' }});

    document.querySelectorAll('.paper-content h2, .paper-content h3').forEach(h => {{
      observer.observe(h);
    }});
  </script>
</body>
</html>
"""

    out_file = "tendon_paper_zh.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Successfully generated {out_file} ({len(full_html)} bytes)")

    out_alias = "paper_zh.html"
    with open(out_alias, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Successfully generated alias {out_alias}")

if __name__ == "__main__":
    build_paper_html()
