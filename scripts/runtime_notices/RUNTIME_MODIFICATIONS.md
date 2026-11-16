# Using a modified LGPL runtime library locally

Kwip ML includes the dynamically linked libquadmath library under LGPL-2.1-or-later. Its license, corresponding GCC source, exact supplier patch and build recipe are supplied in this folder. The applicable open-source licenses govern their covered components. They permit modification and the reverse engineering needed to debug those modifications under their conditions; the application's terms do not remove those rights. The LGPL license text states the complete conditions and warranty limitations.

This guide describes a separate, personally controlled local copy. Keep the App Store installation intact. A modified copy is no longer signed by KWIP and must not be presented as an official KWIP release. It uses a distinct bundle identifier and local settings/container, rather than claiming KWIP's production signing identity. No Apple distribution certificate, private key or account is required for local ad-hoc signing.

## Source and compatibility

Extract `sources/gcc-13.4.0.tar.xz`, apply `sources/gcc-13.4.0-darwin.diff` with `patch -p1`, and use `sources/gcc@13.rb` and `sources/BUILD_CONTEXT.md` for the supplier's arm64 macOS build configuration. The archive includes all three GCC runtime libraries' source and build files. Build your changed library for the same arm64 architecture and libquadmath ABI. Keep its exported symbols and dependencies compatible with the existing SciPy/libgfortran consumers. Merely sharing a filename does not establish ABI compatibility. Building GCC requires a suitable developer SDK, compiler and the dependencies described in that recipe.

## Copy, replace and sign

The following commands are a template for macOS with Apple's developer command-line tools. Replace the three paths with your installation, a **new** destination and your compatible rebuilt library. Do not run them on your original installation. The destination must not already exist; the first check refuses an existing file or directory.

```sh
kwip_original='/Applications/Kwip ML.app'
kwip_modified="$HOME/Applications/Kwip ML Modified.app"
kwip_library='/absolute/path/to/your/libquadmath.0.dylib'
[ ! -e "$kwip_modified" ] && [ ! -L "$kwip_modified" ] || exit 1
/usr/bin/ditto "$kwip_original" "$kwip_modified"
kwip_target="$kwip_modified/Contents/XPCServices/DatabaseSourceService.xpc/Contents/Resources/KwipMLRuntime/python/lib/python3.13/site-packages/scipy/.dylibs/libquadmath.0.dylib"
# Older versions keep the same runtime in the main Resources directory.
if [ ! -f "$kwip_target" ]; then
  kwip_target="$kwip_modified/Contents/Resources/KwipMLRuntime/python/lib/python3.13/site-packages/scipy/.dylibs/libquadmath.0.dylib"
fi
[ -f "$kwip_target" ] || exit 1
/bin/cp "$kwip_library" "$kwip_target"
/usr/bin/install_name_tool -id '@rpath/kwip-user-libquadmath.0.dylib' "$kwip_target"
/usr/bin/codesign --force --sign - --timestamp=none "$kwip_target"
/usr/libexec/PlistBuddy -c 'Set :CFBundleIdentifier info.kwip.KwipML.LocalModification' "$kwip_modified/Contents/Info.plist"
[ ! -e "$kwip_modified.provisionprofile" ] && [ ! -L "$kwip_modified.provisionprofile" ] || exit 1
/bin/mv "$kwip_modified/Contents/embedded.provisionprofile" "$kwip_modified.provisionprofile"
```

Use a different local bundle identifier if you already have another modified copy. The moved provisioning profile belongs to the original distribution identity; it cannot authorize an ad-hoc modified copy. The original app and profile stay intact. The moved copy is evidence only, not a new signing credential.

Create a local `KwipML-local.entitlements` file containing exactly:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>com.apple.security.app-sandbox</key><true/>
<key>com.apple.security.files.user-selected.read-write</key><true/>
<key>com.apple.security.files.bookmarks.app-scope</key><true/>
</dict></plist>
```

For the newer layout, re-seal the containing database helper before the app, preserving its existing entitlements. The historical verification below used the older main-resource layout; a signed replacement check of the newer layout remains part of its release acceptance.

Then sign and verify the local containing app:

```sh
kwip_database_helper="$kwip_modified/Contents/XPCServices/DatabaseSourceService.xpc"
if [ -d "$kwip_database_helper" ]; then
  /usr/bin/codesign --force --sign - --timestamp=none --preserve-metadata=entitlements "$kwip_database_helper"
fi
/usr/bin/codesign --force --sign - --timestamp=none --entitlements KwipML-local.entitlements "$kwip_modified"
/usr/bin/codesign --verify --deep --strict "$kwip_modified"
```

The nested Python interpreter keeps its sandbox-and-inheritance entitlements. Launch the modified **application** from Finder, not its Python executable from a shell; that interpreter requires a sandboxed application parent. In Settings, refresh the runtime status. Open Learn's scikit-learn logistic-regression example, select the estimator, choose Fit estimator, and inspect Results and History. A separate local container means your original app's projects and private file-access grants are not automatically shared; explicitly select any source files you intend to use.

If you change another native library or executable, sign changed nested code before signing the outer app and preserve the executable's appropriate entitlements. Do not use an indiscriminate deep re-sign to replace all nested entitlements. This guide does not require disabling Gatekeeper, System Integrity Protection, certificate checks or the sandbox. Do not bypass a macOS security warning; investigate the local build/signature instead.

## What was verified

On October 9, 2026, a disposable copy of Kwip ML 1.0.0(1) used a separately copied, ABI-identical supplier libquadmath with a deliberately changed loader identity and ad-hoc signature. Its hash differed from the shipped binary. The copied app passed strict deep signature verification. A separate local bundle identifier with the sandbox/file-access entitlements above launched successfully through macOS. Its unchanged bundled interpreter completed the real prepared-data logistic-regression workflow, verified its saved model and portable inference package, and displayed accuracy 1 and log loss 0.06776394708789696 over 16 held-out rows. The worker source checksum validation remained intact. Independent loader enumeration in a host-interpreter check confirmed the replacement file was loaded; SciPy solving and sklearn predictions matched the original fixture.

This verifies the local replacement mechanism on the tested Mac, including the application's sandboxed execution path. It does not promise that every modified library is ABI-compatible, reproduce a numerical-code change built from source, validate another Mac, or test a future Apple-processed store download. The initial attempt to keep production team/application-identifier entitlements with an ad-hoc signature failed to launch; the separate local identity is required by the tested procedure. Apple validation and release approval remain separate from this mechanism.
