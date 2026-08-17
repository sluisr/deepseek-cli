/**
 * @license
 * Copyright 2026 sluisr
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

const ASKPASS_SCRIPT_CONTENT = `#!/usr/bin/env node
const fs = require('fs');
const tty = require('tty');

const promptMsg = process.argv[2] || 'Password: ';

try {
  const ttyFdIn = fs.openSync('/dev/tty', 'r');
  const ttyFdOut = fs.openSync('/dev/tty', 'w');

  const inStream = new tty.ReadStream(ttyFdIn);
  const outStream = new tty.WriteStream(ttyFdOut);

  outStream.write(\`\\n\\x1b[1;36m🔑 [DeepSeek CLI]\\x1b[0m \\x1b[1m\${promptMsg}\\x1b[0m \`);

  inStream.setRawMode(true);
  inStream.resume();
  inStream.setEncoding('utf-8');

  let password = '';

  inStream.on('data', (chunk) => {
    for (let i = 0; i < chunk.length; i++) {
      const char = chunk[i];
      if (char === '\\n' || char === '\\r' || char === '\\u0004') {
        outStream.write('\\n');
        try {
          inStream.setRawMode(false);
          inStream.pause();
        } catch {}
        process.stdout.write(password);
        process.exit(0);
      } else if (char === '\\u0003') {
        outStream.write('^C\\n');
        try {
          inStream.setRawMode(false);
          inStream.pause();
        } catch {}
        process.exit(130);
      } else if (char === '\\u0008' || char === '\\x7f') {
        if (password.length > 0) {
          password = password.slice(0, -1);
        }
      } else {
        password += char;
      }
    }
  });
} catch (err) {
  process.exit(1);
}
`;

let cachedAskPassPath: string | null = null;

export function getOrInitAskPassScript(): string {
  if (cachedAskPassPath && fs.existsSync(cachedAskPassPath)) {
    return cachedAskPassPath;
  }

  const dir = path.join(
    process.env['HOME'] || process.env['USERPROFILE'] || os.tmpdir(),
    '.deepseek',
  );

  try {
    fs.mkdirSync(dir, { recursive: true });
    const scriptPath = path.join(dir, 'askpass.cjs');
    fs.writeFileSync(scriptPath, ASKPASS_SCRIPT_CONTENT, {
      mode: 0o755,
      encoding: 'utf-8',
    });
    cachedAskPassPath = scriptPath;
    return scriptPath;
  } catch {
    const fallbackPath = path.join(os.tmpdir(), 'deepseek_askpass.cjs');
    fs.writeFileSync(fallbackPath, ASKPASS_SCRIPT_CONTENT, {
      mode: 0o755,
      encoding: 'utf-8',
    });
    cachedAskPassPath = fallbackPath;
    return fallbackPath;
  }
}
