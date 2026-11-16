import XCTest
import SwiftUI
import AppKit
import KwipMLCore
@testable import KwipMLApp

final class AppearanceThemeTests: XCTestCase {
    func testExplicitModeOverridesSystemAndSystemFollowsProvidedScheme() {
        XCTAssertEqual(AppearanceThemePolicy(preferences:.init(mode:.dark),colorScheme:.light,increasedContrast:false,reduceTransparency:false).colorScheme,.dark)
        XCTAssertEqual(AppearanceThemePolicy(preferences:.init(mode:.light),colorScheme:.dark,increasedContrast:false,reduceTransparency:false).colorScheme,.light)
        for scheme in [ColorScheme.dark,.light] {
            XCTAssertEqual(AppearanceThemePolicy(preferences:.init(mode:.system),colorScheme:scheme,increasedContrast:false,reduceTransparency:false).colorScheme,scheme)
        }
    }
    func testAccessibilityOverridesEffectiveChromeWithoutChangingSavedChoice() {
        let preferences=AppearancePreferences(mode:.system,palette:.teal,transparency:.subtle)
        for contrast in [false,true] {
            for reduced in [false,true] {
                let policy=AppearanceThemePolicy(preferences:preferences,colorScheme:.dark,increasedContrast:contrast,reduceTransparency:reduced)
                XCTAssertEqual(policy.chromeOpaque,contrast || reduced)
                XCTAssertEqual(policy.preferences,preferences)
            }
        }
        XCTAssertTrue(AppearanceThemePolicy(preferences:.factory,colorScheme:.dark,increasedContrast:false,reduceTransparency:false).chromeOpaque)
    }
    func testEveryEnabledTextAndProminentLabelHasReadableNativeContrast() throws {
        for scheme in [ColorScheme.dark,.light] {
            for accent in AppearancePreferences.Palette.allCases {
                for increased in [false,true] {
                    let policy=AppearanceThemePolicy(preferences:.init(mode:.system,palette:accent),colorScheme:scheme,increasedContrast:increased,reduceTransparency:false)
                    let palette=policy.native
                    for foreground in [palette.text,palette.muted] {
                        for background in [palette.background,palette.panel] {
                            XCTAssertGreaterThanOrEqual(try ratio(foreground,background),4.5,"\(scheme), \(accent), increased=\(increased)")
                        }
                    }
                    XCTAssertGreaterThanOrEqual(try ratio(palette.accentForeground,palette.accent),4.5)
                    for token in [palette.keyword,palette.string,palette.comment,palette.number,palette.name] {
                        XCTAssertGreaterThanOrEqual(try ratio(token,palette.background),4.5)
                    }
                }
            }
        }
    }
    func testFingerprintChangesForRenderedSyntaxChoicesButNotUnrelatedChrome() {
        func make(_ mode:AppearancePreferences.Mode = .dark,_ palette:AppearancePreferences.Palette = .rose,_ contrast:Bool = false,_ transparency:AppearancePreferences.Transparency = .opaque)->AppearanceThemePolicy {
            .init(preferences:.init(mode:mode,palette:palette,transparency:transparency),colorScheme:.dark,increasedContrast:contrast,reduceTransparency:false)
        }
        XCTAssertNotEqual(make().syntaxPaletteFingerprint,make(.light).syntaxPaletteFingerprint)
        XCTAssertNotEqual(make().syntaxPaletteFingerprint,make(.dark,.teal).syntaxPaletteFingerprint)
        XCTAssertNotEqual(make().syntaxPaletteFingerprint,make(.dark,.rose,true).syntaxPaletteFingerprint)
        XCTAssertEqual(make().syntaxPaletteFingerprint,make(.dark,.rose,false,.subtle).syntaxPaletteFingerprint)
    }
    private func ratio(_ foreground:NSColor,_ background:NSColor)throws->Double {
        func luminance(_ color:NSColor)throws->Double {
            let value=try XCTUnwrap(color.usingColorSpace(.sRGB))
            XCTAssertEqual(value.alphaComponent,1,accuracy:0.000001,"Contrast is measured on resolved opaque colors")
            func channel(_ value:CGFloat)->Double {let v=Double(value);return v<=0.04045 ? v/12.92:pow((v+0.055)/1.055,2.4)}
            return channel(value.redComponent)*0.2126+channel(value.greenComponent)*0.7152+channel(value.blueComponent)*0.0722
        }
        let a=try luminance(foreground),b=try luminance(background)
        return (max(a,b)+0.05)/(min(a,b)+0.05)
    }
}
