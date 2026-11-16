import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'sparkle_build.py'
spec=importlib.util.spec_from_file_location('sparkle_build',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_recipe_requires_sandboxed_custom_downloader_and_installer_bridges(tmp_path):
    commands=module.commands(tmp_path/'source',tmp_path/'derived',tmp_path/'Sparkle.xcframework')
    build=commands[0]
    for option in ['ARCHS=arm64','ONLY_ACTIVE_ARCH=YES','MACOSX_DEPLOYMENT_TARGET=14.0',
      'DOWNLOADER_SANDBOXED_ENTITLEMENTS=Downloader/Downloader.entitlements',
      'XPC_SERVICE_BUNDLE_ID_PREFIX=info.kwip.KwipML.Sparkle',
      'SPARKLE_EMBED_INSTALLER_CONNECTION_XPC_SERVICE=1','SPARKLE_EMBED_INSTALLER_STATUS_XPC_SERVICE=1','CODE_SIGNING_ALLOWED=NO']:
        assert option in build
    assert commands[1][1]=='-create-xcframework'


def test_tree_digest_covers_bytes_names_and_internal_symlink_targets(tmp_path):
    root=tmp_path/'framework';root.mkdir();(root/'binary').write_bytes(b'one');(root/'alias').symlink_to('binary')
    original=module.tree_digest(root)
    assert module.tree_digest(root)==original
    (root/'binary').write_bytes(b'two');assert module.tree_digest(root)!=original
    (root/'binary').write_bytes(b'one');(root/'alias').unlink();(root/'other').write_bytes(b'one');(root/'alias').symlink_to('other')
    assert module.tree_digest(root)!=original


def test_tree_digest_refuses_external_and_broken_links(tmp_path):
    root=tmp_path/'framework';root.mkdir();other=tmp_path/'outside';other.write_bytes(b'secret')
    (root/'escape').symlink_to(other)
    with pytest.raises(ValueError):module.tree_digest(root)
    (root/'escape').unlink();(root/'broken').symlink_to('missing')
    with pytest.raises(ValueError):module.tree_digest(root)


def test_output_must_be_new_ignored_local_child_without_symlink_parents(tmp_path):
    root=tmp_path/'repo';root.mkdir();local=root/'.local';local.mkdir()
    assert module.output_path(root,local/'sparkle')==local/'sparkle'
    for bad in [local,root/'tracked',tmp_path/'outside']:
        with pytest.raises(ValueError):module.output_path(root,bad)
    existing=local/'existing';existing.mkdir()
    with pytest.raises(ValueError):module.output_path(root,existing)
    link=local/'link';link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):module.output_path(root,link/'sparkle')
