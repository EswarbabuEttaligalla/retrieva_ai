import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
  url: 'http://localhost:5173',
  pretendToBeVisual: true,
});

globalThis.window = dom.window;
globalThis.document = dom.window.document;

const userAgent = dom.window.navigator.userAgent || 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

Object.defineProperty(globalThis, 'navigator', {
  value: {
    userAgent,
    clipboard: {
      writeText: async () => {},
      readText: async () => '',
    },
  },
  configurable: true,
  writable: true,
});

globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.HTMLElement.prototype.scrollIntoView = function() {};
globalThis.HTMLInputElement = dom.window.HTMLInputElement;
globalThis.Node = dom.window.Node;
globalThis.File = dom.window.File;
globalThis.Blob = dom.window.Blob;
globalThis.FormData = dom.window.FormData;
globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
