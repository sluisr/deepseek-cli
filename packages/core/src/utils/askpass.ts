/**
 * @license
 * Copyright 2026 sluisr
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

/**
 * Session-scoped sudo password (in-memory only, never persisted to disk).
 * Set via `$sudo:PASSWORD` prefix in the CLI prompt.
 */
let sessionSudoPassword: string | null = null;

/**
 * Store the sudo password in session memory.
 * The password is NEVER sent to the DeepSeek API, written to disk, or logged.
 */
export function setSudoPassword(password: string): void {
  sessionSudoPassword = password;
}

/**
 * Retrieve the stored sudo password, or null if none has been set.
 */
export function getSudoPassword(): string | null {
  return sessionSudoPassword;
}

/**
 * Clear the stored sudo password (called on session exit).
 */
export function clearSudoPassword(): void {
  sessionSudoPassword = null;
}

const ASKPASS_SCRIPT_CONTENT = `#!/usr/bin/env node
const fs = require('fs');
const tty = require('tty');

const promptMsg = process.argv[2] || 'Password: ';
const NL = String.fromCharCode(10);

// If DEEPSEEK_SUDO_PASSWORD is set, echo it immediately (non-interactive mode)
if (process.env.DEEPSEEK_SUDO_PASSWORD) {
  try {
    fs.writeSync(1, process.env.DEEPSEEK_SUDO_PASSWORD + NL);
  } catch {
    process.stdout.write(process.env.DEEPSEEK_SUDO_PASSWORD + NL);
  }
  process.exit(0);
}

// Otherwise, fall back to interactive TTY prompt
try {
  const ttyFdIn = fs.openSync('/dev/tty', 'r');
  const ttyFdOut = fs.openSync('/dev/tty', 'w');

  const inStream = new tty.ReadStream(ttyFdIn);
  const outStream = new tty.WriteStream(ttyFdOut);

  outStream.write(NL + '\\x1b[1;36m🔑 [DeepSeek CLI]\\x1b[0m \\x1b[1m' + promptMsg + '\\x1b[0m ');

  inStream.setRawMode(true);
  inStream.resume();
  inStream.setEncoding('utf-8');

  let password = '';

  inStream.on('data', (chunk) => {
    for (let i = 0; i < chunk.length; i++) {
      const char = chunk[i];
      if (char === String.fromCharCode(10) || char === String.fromCharCode(13) || char === String.fromCharCode(4)) {
        outStream.write(NL);
        try {
          inStream.setRawMode(false);
          inStream.pause();
        } catch {}
        try {
          fs.writeSync(1, password + NL);
        } catch {
          process.stdout.write(password + NL);
        }
        process.exit(0);
      } else if (char === String.fromCharCode(3)) {
        outStream.write('^C' + NL);
        try {
          inStream.setRawMode(false);
          inStream.pause();
        } catch {}
        process.exit(130);
      } else if (char === String.fromCharCode(8) || char === String.fromCharCode(127)) {
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

export function getOrInitAskPassScript(): string {
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
    return scriptPath;
  } catch {
    const fallbackPath = path.join(os.tmpdir(), 'deepseek_askpass.cjs');
    fs.writeFileSync(fallbackPath, ASKPASS_SCRIPT_CONTENT, {
      mode: 0o755,
      encoding: 'utf-8',
    });
    return fallbackPath;
  }
}
