# -*- mode: python ; coding: utf-8 -*-
import os.path
block_cipher = None
a = Analysis(['src/flatting_server.py'],
             ## pyinstaller iheartla.spec must be run from
             ## ???
             pathex=[os.path.abspath(os.getcwd())],
             binaries=[],
             datas=[('src/flatting/checkpoints','checkpoints')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='flatting_server',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon='src/flatting/resources/flatting.ico' )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='flatting_server')
app = BUNDLE(coll,
             name='flatting_server.app',
             icon='src/flatting/resources/flatting.icns',
             bundle_identifier=None,
             info_plist={'NSHighResolutionCapable': 'True'}
             )
