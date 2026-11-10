"""BUILD-V3FIX: two of the three windows the empty-text 400 stopped, frozen for tests/test_empty_text.py.

Each was read once, read-only, from the stopped run's L ledger (opened with mode=ro&immutable=1): the
window replayed to just before the run's first failed `call` event (no request body is recorded for a
call the API refused, so the window is what is kept). Window positions are as in the ledger, the
system prompt (position 0) left out: it is L's prompt `prompt_version` from code, byte for byte (same
sha256), and the tool definitions are L's CTX_TOOLS + CORPUS_TOOLS (same tools_hash). Each segment is
kept as the text the request carried for it (a span's alias header included, a fold as its summary),
an assistant tool turn as its content blocks. `api_error` is the first 400 the API returned for that
call (runs/v3/grid.log); `empty` is the position of the empty user segment L appended, `frozen` and
`prev_last` the BP2 and BP3 positions of the call's build. The third stopped run
(v3-grid-haiku-forced-r100-e6-10) has the same shape as the forced one (messages.4 instead of 6).
"""

CASES = {'v3-grid-haiku-allowed-r10-e6-10': {'ledger': 'runs/v3-grid-haiku-allowed-r10-e6-10/L/episode-06.sqlite',
                                     'call_id': 'call_171d78ce99b4ac78',
                                     'call_seq': 1452,
                                     'prompt_version': 'clp-v3-allowed',
                                     'max_tokens': 1024,
                                     'ttl': '5m',
                                     'thinking': {'type': 'disabled'},
                                     'frozen': None,
                                     'prev_last': 10,
                                     'empty': 12,
                                     'api_error': 'messages: text content blocks must be non-empty',
                                     'segments': [{'role': 'user', 'text': '[fold @PH]\n- (@PJ) earlier context: 1 fold [@P5]'},
                                                  {'role': 'user',
                                                   'text': 'Which exception classes does `compression.zstd._zstdfile.ZstdFile.__init__` '
                                                           'raise directly (in its own body)?'},
                                                  {'role': 'user', 'text': '[fold @PD]\n- (@PE) earlier context: 1 fold [@NX]'},
                                                  {'role': 'assistant',
                                                   'blocks': [{'text': 'I have only part of the `__init__` body so far. The read returned '
                                                                       "lines 2829-3477 and a `close` method, which doesn't match the "
                                                                       'requested line range, and the earlier fold text described the '
                                                                       'start of `__init__` only through line 77. I can see the '
                                                                       "`TypeError` raised for an invalid `file` argument, but I haven't "
                                                                       "confirmed the rest of `__init__`, so I can't give a complete list "
                                                                       'of directly raised exceptions yet.\n'
                                                                       '\n'
                                                                       'What I can confirm from the verbatim text:\n'
                                                                       "- `__init__` raises `TypeError('file must be a file-like object or "
                                                                       "a str, bytes, or PathLike object')` when `file` is neither a "
                                                                       'usable file-like object nor a path (quoted from @NK).\n'
                                                                       '\n'
                                                                       "I haven't seen the complete body (e.g., any `ValueError` checks "
                                                                       "for mode or level arguments), so I'm not able to state that this "
                                                                       'is the full set. Let me check the earlier section of `__init__` to '
                                                                       'finish.',
                                                               'type': 'text'},
                                                              {'id': 'toolu_018ie2uknwb3XvHZnHUYsrsT',
                                                               'input': {'end_line': 77,
                                                                         'path': 'Lib/compression/zstd/_zstdfile.py',
                                                                         'start_line': 50},
                                                               'name': 'corpus_read',
                                                               'type': 'tool_use'}]},
                                                  {'role': 'tool',
                                                   'text': '[@N9 Lib/compression/zstd/_zstdfile.py#1619-2828]\n'
                                                           '        *zstd_dict* is an optional ZstdDict object, a pre-trained Zstandard\n'
                                                           '        dictionary. See train_dict() to train ZstdDict on sample data.\n'
                                                           '        """\n'
                                                           '        self._fp = None\n'
                                                           '        self._close_fp = False\n'
                                                           '        self._mode = _MODE_CLOSED\n'
                                                           '        self._buffer = None\n'
                                                           '\n'
                                                           '        if not isinstance(mode, str):\n'
                                                           "            raise ValueError('mode must be a str')\n"
                                                           '        if options is not None and not isinstance(options, dict):\n'
                                                           "            raise TypeError('options must be a dict or None')\n"
                                                           "        mode = mode.removesuffix('b')  # handle rb, wb, xb, ab\n"
                                                           "        if mode == 'r':\n"
                                                           '            if level is not None:\n'
                                                           "                raise TypeError('level is illegal in read mode')\n"
                                                           '            self._mode = _MODE_READ\n'
                                                           "        elif mode in {'w', 'a', 'x'}:\n"
                                                           '            if level is not None and not isinstance(level, int):\n'
                                                           "                raise TypeError('level must be int or None')\n"
                                                           '            self._mode = _MODE_WRITE\n'
                                                           '            self._compressor = ZstdCompressor(level=level, options=options,\n'
                                                           '                                              zstd_dict=zstd_dict)\n'
                                                           '            self._pos = 0\n'
                                                           '        else:\n'
                                                           "            raise ValueError(f'Invalid mode: {mode!r}')\n"
                                                           '\n'
                                                           '        if isinstance(file, (str, bytes, PathLike)):',
                                                   'tool_use_id': 'toolu_018ie2uknwb3XvHZnHUYsrsT'},
                                                  {'role': 'tool',
                                                   'text': 'read Lib/compression/zstd/_zstdfile.py:50-77 as @N9 (text)',
                                                   'tool_use_id': 'toolu_018ie2uknwb3XvHZnHUYsrsT'},
                                                  {'role': 'user', 'text': '[fold @PK]\n- (@PM) earlier context: 1 fold [@PF]'},
                                                  {'role': 'assistant',
                                                   'blocks': [{'text': "I'll refetch @NK so the claim rests on verbatim text.",
                                                               'type': 'text'},
                                                              {'id': 'toolu_01EhY2WyKtKuM2M3vadCc4gV',
                                                               'input': {'refs': ['@NK']},
                                                               'name': 'ctx_refetch',
                                                               'type': 'tool_use'}]},
                                                  {'role': 'tool',
                                                   'text': '[@NK Lib/compression/zstd/_zstdfile.py#2829-3477]\n'
                                                           "            self._fp = io.open(file, f'{mode}b')\n"
                                                           '            self._close_fp = True\n'
                                                           "        elif ((mode == 'r' and hasattr(file, 'read'))\n"
                                                           "                or (mode != 'r' and hasattr(file, 'write'))):\n"
                                                           '            self._fp = file\n'
                                                           '        else:\n'
                                                           "            raise TypeError('file must be a file-like object '\n"
                                                           "                            'or a str, bytes, or PathLike object')\n"
                                                           '\n'
                                                           '        if self._mode == _MODE_READ:\n'
                                                           '            raw = _streams.DecompressReader(\n'
                                                           '                self._fp,\n'
                                                           '                ZstdDecompressor,\n'
                                                           '                zstd_dict=zstd_dict,\n'
                                                           '                options=options,\n'
                                                           '            )\n'
                                                           '            self._buffer = io.BufferedReader(raw)\n',
                                                   'tool_use_id': 'toolu_01EhY2WyKtKuM2M3vadCc4gV'},
                                                  {'role': 'tool', 'text': '@NK OK', 'tool_use_id': 'toolu_01EhY2WyKtKuM2M3vadCc4gV'},
                                                  {'role': 'user',
                                                   'text': '[fold @PP]\n'
                                                           '- (@PQ) assistant message: The refetched text confirms the `TypeError` for the '
                                                           'file argument. Here is what the verbatim spans show for `ZstdFile.__ [@PB]'},
                                                  {'role': 'user', 'text': ''},
                                                  {'role': 'user', 'text': 'What is the current auth realm of service alcor-router?'}]},
 'v3-grid-haiku-forced-r10-e6-10': {'ledger': 'runs/v3-grid-haiku-forced-r10-e6-10/L/episode-07.sqlite',
                                    'call_id': 'call_d07a4398ec9da391',
                                    'call_seq': 312,
                                    'prompt_version': 'clp-v3-forced',
                                    'max_tokens': 1024,
                                    'ttl': '5m',
                                    'thinking': {'type': 'disabled'},
                                    'frozen': 1,
                                    'prev_last': 11,
                                    'empty': 13,
                                    'api_error': 'messages.6.content.0.text: cache_control cannot be set for empty text blocks',
                                    'segments': [{'role': 'user', 'text': '[fold @11]\n- (@12) earlier context: 1 fold [@X]'},
                                                 {'role': 'user', 'text': '[fold @20]\n- (@21) earlier fold @1A (16 sentences) [@1A]'},
                                                 {'role': 'user', 'text': '[fold @18]\n- (@19) earlier context: 1 fold [@14]'},
                                                 {'role': 'user',
                                                  'text': 'Which exception classes does `dbm.sqlite3._Database._delta_timber` raise '
                                                          'directly (in its own body)?'},
                                                 {'role': 'user',
                                                  'text': '[fold @2J]\n'
                                                          "- (@2K) assistant message: I'll search for the function body first. [tool_use "
                                                          'corpus_search] [@16]\n'
                                                          '- (@2M) tool message: Lib/dbm/sqlite3.py:97-133  raise KeyError(key) [@17]\n'
                                                          '- (@2N) earlier fold @26 (7 sentences) [@26]\n'
                                                          '- (@2P) assistant message: The module defines `class error(OSError)` at '
                                                          '`Lib/dbm/sqlite3.py:21-28`, so `error` is a module-level subclass of `OSErr '
                                                          '[@24]\n'
                                                          '- (@2Q) user message: Verification: claim 1 was not accepted: ref '
                                                          '`Lib/dbm/sqlite3.py#_Database._delta_timber` and `Lib/dbm/sqlite3.py#_Databa '
                                                          '[@25]\n'
                                                          '- (@2R) assistant message: I need to correct my previous answer. The '
                                                          "verification flagged my citations, and looking back at the material, I can't c "
                                                          '[@2E]'},
                                                 {'role': 'user',
                                                  'text': 'In this condition you must give a best-guess answer, with claims and quotes.'},
                                                 {'role': 'assistant',
                                                  'text': "My best guess is `error`, the module's own `OSError` subclass, but I hold no "
                                                          'verbatim span showing the body of `_delta_timber`, so I cannot give a '
                                                          'quote-backed claim for it.\n'
                                                          '\n'
                                                          'ANSWER: error\n'
                                                          '<claims>[{"text": "Best guess: `dbm.sqlite3._Database._delta_timber` raises the '
                                                          'module-level class `error` directly; not confirmed by a verbatim body span.", '
                                                          '"citations": [{"ref": "@1Y", "quote": "def _delta_timber(self, *args, '
                                                          '**kwargs):"}]}]</claims>'},
                                                 {'role': 'user',
                                                  'text': 'Verification: claim 1 relies on @1Y which you no longer hold verbatim '
                                                          '(summary_only). Call ctx_refetch on them and answer again.'},
                                                 {'role': 'assistant',
                                                  'blocks': [{'text': "I'll refetch @1Y as requested.", 'type': 'text'},
                                                             {'id': 'toolu_01LkNZjzsvyD7ho3g9YHPa17',
                                                              'input': {'refs': ['@1Y']},
                                                              'name': 'ctx_refetch',
                                                              'type': 'tool_use'}]},
                                                 {'role': 'tool',
                                                  'text': '[@1Y Lib/dbm/sqlite3.py#_Database._delta_timber]\n'
                                                          'def _delta_timber(self, *args, **kwargs):\n'
                                                          '        if not self._cx:\n'
                                                          '            raise error(_ERR_CLOSED)\n'
                                                          '        try:\n'
                                                          '            return closing(self._cx.execute(*args, **kwargs))\n'
                                                          '        except sqlite3.Error as exc:\n'
                                                          '            raise error(str(exc))',
                                                  'tool_use_id': 'toolu_01LkNZjzsvyD7ho3g9YHPa17'},
                                                 {'role': 'tool', 'text': '@1Y OK', 'tool_use_id': 'toolu_01LkNZjzsvyD7ho3g9YHPa17'},
                                                 {'role': 'assistant',
                                                  'text': 'The refetched body shows `_delta_timber` raises `error` directly in two places, '
                                                          'so the earlier `KeyError` line at `Lib/dbm/sqlite3.py:97-133` belongs to a '
                                                          "different function and does not apply here. I'll rely on this refetched span, "
                                                          'which I now hold verbatim.\n'
                                                          '\n'
                                                          'ANSWER: error\n'
                                                          '<claims>[{"text": "`dbm.sqlite3._Database._delta_timber` raises `error` '
                                                          'directly in its own body (twice: when the connection is closed, and when '
                                                          '`sqlite3.Error` is caught).", "citations": [{"ref": "@1Y", "quote": "raise '
                                                          'error(_ERR_CLOSED)"}, {"ref": "@1Y", "quote": "raise '
                                                          'error(str(exc))"}]}]</claims>'},
                                                 {'role': 'user', 'text': ''}]}}
