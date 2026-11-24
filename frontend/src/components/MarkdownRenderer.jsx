import React, { useState } from 'react';
import { Copy, Check, Terminal } from 'lucide-react';

/**
 * Fenced code block with copy action and language label.
 */
function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3 rounded-lg overflow-hidden border border-slate-700/80 bg-slate-900/90 shadow-md">
      <div className="flex items-center justify-between px-3 py-1.5 bg-slate-800/80 border-b border-slate-700/60 text-xs font-mono text-slate-300">
        <div className="flex items-center space-x-1.5">
          <Terminal className="w-3.5 h-3.5 text-brand-400" />
          <span>{language || 'code'}</span>
        </div>
        <button
          onClick={handleCopy}
          className="inline-flex items-center space-x-1 px-2 py-0.5 rounded hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-400" />
              <span className="text-emerald-400 text-[11px]">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span className="text-[11px]">Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 text-xs text-slate-200 font-mono overflow-x-auto leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
}

/**
 * Parses inline formatting: bold, italic, code, and [Source N] citation pills.
 */
function renderInlineFormatting(text, onCitationClick) {
  if (!text) return null;

  // Regex tokens:
  // 1: Fenced inline code `code`
  // 2: Bold **text**
  // 3: Italic *text*
  // 4: Citation tags: [Source N] or [Source N: filename, p. X]
  // 5: Safe external links: [label](https://...) or [label](http://...)
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[Source\s+\d+[^\]]*\]|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;
  const parts = text.split(pattern);

  return parts.map((part, idx) => {
    if (!part) return null;

    // Inline code
    if (part.startsWith('`') && part.endsWith('`') && part.length > 1) {
      return (
        <code
          key={idx}
          className="px-1.5 py-0.5 mx-0.5 text-xs font-mono rounded bg-slate-800 text-brand-300 border border-slate-700/60"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    // Bold
    if (part.startsWith('**') && part.endsWith('**') && part.length > 3) {
      return (
        <strong key={idx} className="font-semibold text-slate-100">
          {renderInlineFormatting(part.slice(2, -2), onCitationClick)}
        </strong>
      );
    }

    // Italic
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return (
        <em key={idx} className="italic text-slate-300">
          {renderInlineFormatting(part.slice(1, -1), onCitationClick)}
        </em>
      );
    }

    // Citation pill: [Source N ...]
    const citationMatch = part.match(/^\[Source\s+(\d+)([^\]]*)\]$/);
    if (citationMatch) {
      const sourceNum = citationMatch[1];
      const sourceId = `Source ${sourceNum}`;
      return (
        <button
          key={idx}
          type="button"
          onClick={() => onCitationClick && onCitationClick(sourceId)}
          className="inline-flex items-center px-1.5 py-0.2 mx-1 my-0.5 text-[11px] font-semibold font-mono rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500/30 hover:border-indigo-400 transition cursor-pointer align-baseline"
          title={`Click to inspect ${sourceId}`}
        >
          {part}
        </button>
      );
    }

    // Safe external link: [label](https://...) or [label](http://...)
    const linkMatch = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
    if (linkMatch) {
      const label = linkMatch[1];
      const url = linkMatch[2];
      return (
        <a
          key={idx}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-brand-400 hover:text-brand-300 underline font-medium"
        >
          {label}
        </a>
      );
    }

    return part;
  });
}

/**
 * Production-grade, XSS-safe Markdown Renderer.
 * Supports headings, code blocks, lists, blockquotes, inline formatting, and citation pills.
 */
export default function MarkdownRenderer({ content, onCitationClick }) {
  if (!content) return null;

  const lines = content.split('\n');
  const elements = [];
  let inCodeBlock = false;
  let codeBuffer = [];
  let codeLang = '';
  let inList = false;
  let listItems = [];
  let listType = 'ul'; // 'ul' or 'ol'

  const flushList = () => {
    if (listItems.length > 0) {
      if (listType === 'ol') {
        elements.push(
          <ol key={`ol-${elements.length}`} className="my-2 ml-5 space-y-1 list-decimal text-slate-300 text-sm">
            {listItems.map((item, i) => (
              <li key={i}>{renderInlineFormatting(item, onCitationClick)}</li>
            ))}
          </ol>
        );
      } else {
        elements.push(
          <ul key={`ul-${elements.length}`} className="my-2 ml-5 space-y-1 list-disc text-slate-300 text-sm">
            {listItems.map((item, i) => (
              <li key={i}>{renderInlineFormatting(item, onCitationClick)}</li>
            ))}
          </ul>
        );
      }
      listItems = [];
      inList = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Fenced code block toggle
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        // Closing code block
        elements.push(
          <CodeBlock
            key={`code-${elements.length}`}
            language={codeLang}
            code={codeBuffer.join('\n')}
          />
        );
        codeBuffer = [];
        codeLang = '';
        inCodeBlock = false;
      } else {
        // Opening code block
        flushList();
        codeLang = line.trim().slice(3).trim();
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(line);
      continue;
    }

    // Unordered List Items (- or * or +)
    const ulMatch = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ulMatch) {
      if (inList && listType !== 'ul') {
        flushList();
      }
      inList = true;
      listType = 'ul';
      listItems.push(ulMatch[1]);
      continue;
    }

    // Ordered List Items (1. or 2.)
    const olMatch = line.match(/^\s*(\d+)\.\s+(.*)$/);
    if (olMatch) {
      if (inList && listType !== 'ol') {
        flushList();
      }
      inList = true;
      listType = 'ol';
      listItems.push(olMatch[2]);
      continue;
    }

    // Flush any pending list if line is not a list item
    flushList();

    // Empty lines
    if (!line.trim()) {
      continue;
    }

    // Headings
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={`h3-${i}`} className="mt-4 mb-2 text-base font-semibold text-white tracking-tight">
          {renderInlineFormatting(line.slice(4), onCitationClick)}
        </h3>
      );
      continue;
    }
    if (line.startsWith('## ')) {
      elements.push(
        <h2 key={`h2-${i}`} className="mt-5 mb-2 text-lg font-bold text-white tracking-tight">
          {renderInlineFormatting(line.slice(3), onCitationClick)}
        </h2>
      );
      continue;
    }
    if (line.startsWith('# ')) {
      elements.push(
        <h1 key={`h1-${i}`} className="mt-5 mb-3 text-xl font-extrabold text-white tracking-tight">
          {renderInlineFormatting(line.slice(2), onCitationClick)}
        </h1>
      );
      continue;
    }

    // Blockquote
    if (line.startsWith('> ')) {
      elements.push(
        <blockquote
          key={`quote-${i}`}
          className="my-2.5 pl-3.5 py-1 border-l-2 border-brand-500 text-slate-300 italic text-sm bg-slate-900/40 rounded-r"
        >
          {renderInlineFormatting(line.slice(2), onCitationClick)}
        </blockquote>
      );
      continue;
    }

    // Standard Paragraph
    elements.push(
      <p key={`p-${i}`} className="my-2 text-sm text-slate-200 leading-relaxed">
        {renderInlineFormatting(line, onCitationClick)}
      </p>
    );
  }

  // Flush remaining blocks
  if (inCodeBlock && codeBuffer.length > 0) {
    elements.push(
      <CodeBlock
        key={`code-${elements.length}`}
        language={codeLang}
        code={codeBuffer.join('\n')}
      />
    );
  }
  flushList();

  return <div className="space-y-1">{elements}</div>;
}
