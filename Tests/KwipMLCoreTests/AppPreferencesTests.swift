import XCTest
@testable import KwipMLCore

final class AppPreferencesTests: XCTestCase {
    func testScrollZoomSensitivityMigratesAndPersistsWithoutChangingCanvasDefaults() throws {
        let dir=try directory();defer {try? FileManager.default.removeItem(at:dir)}
        let store=AppPreferencesStore(fileURL:dir.appendingPathComponent("app-preferences.json"))
        var prefs=try store.load();XCTAssertEqual(prefs.effectiveCanvasZoomSensitivity,1)
        let canvas=prefs.canvasDefaults
        prefs.canvasZoomSensitivity=0.5;try store.save(prefs)
        XCTAssertEqual(try store.load().effectiveCanvasZoomSensitivity,0.5)
        XCTAssertEqual(try store.load().canvasDefaults,canvas)
        prefs.canvasZoomSensitivity=0;XCTAssertThrowsError(try store.save(prefs))
        prefs.canvasZoomSensitivity=3;XCTAssertThrowsError(try store.save(prefs))
        XCTAssertEqual(try store.load().effectiveCanvasZoomSensitivity,0.5)
    }
    func testPresentationDefaultsDoNotTurnInitialViewportIntoSavedNavigation() {
        var state = CanvasViewState(presentation: .init(flowDirection: .topToBottom, nodeDetail: .large))
        XCTAssertTrue(state.isInitialViewport)
        state.scrollX = 12; XCTAssertFalse(state.isInitialViewport)
        state.scrollX = 0; state.zoom = 0.7; XCTAssertFalse(state.isInitialViewport)
        state.zoom = 0.65; state.selectedNodeIDs = ["node"]; XCTAssertFalse(state.isInitialViewport)
        state.selectedNodeIDs = []; state.selectedEdgeIDs = ["edge"]; XCTAssertFalse(state.isInitialViewport)
    }
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func testLegacyMigrationAndIndependentResetsPreserveOtherSettings() throws {
        let dir=try directory();defer {try? FileManager.default.removeItem(at:dir)}
        let file=dir.appendingPathComponent("app-preferences.json"),store=AppPreferencesStore(fileURL:file)
        let old=Data(#"{"schemaVersion":1,"canvasDefaults":{"flowDirection":"topToBottom","nodeDetail":"small"}}"#.utf8)
        try old.write(to:file)
        var prefs=try store.load()
        XCTAssertNil(prefs.shortcutOverrides)
        prefs.shortcutOverrides=["learningLibrary":ShortcutChord(key:"k",shift:true)]
        try store.save(prefs)
        XCTAssertEqual(try store.resetCanvasDefaults().shortcutOverrides,prefs.shortcutOverrides)
        prefs.canvasDefaults = .init(flowDirection:.topToBottom,nodeDetail:.small)
        try store.save(prefs)
        let reset=try store.resetShortcuts()
        XCTAssertNil(reset.shortcutOverrides)
        XCTAssertEqual(reset.canvasDefaults,prefs.canvasDefaults)
        prefs.shortcutOverrides=["learningLibrary":ShortcutChord(key:"s")]
        XCTAssertThrowsError(try store.save(prefs))
        XCTAssertEqual(try store.load(),reset)
    }

    func testMissingDefaultsRoundTripAndScopedReset() throws {
        let dir = try directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("app-preferences.json")
        let store = AppPreferencesStore(fileURL: file)
        XCTAssertEqual(try store.load(), .factory)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        let project = dir.appendingPathComponent("project.json")
        let saved = Data("saved project and arrangement sentinel".utf8)
        try saved.write(to: project)
        let custom = AppPreferences(canvasDefaults: .init(flowDirection: .topToBottom, nodeDetail: .small))
        try store.save(custom)
        XCTAssertEqual(try AppPreferencesStore(fileURL: file).load(), custom)
        XCTAssertEqual(try store.resetCanvasDefaults(), .factory)
        XCTAssertEqual(try store.load(), .factory)
        XCTAssertEqual(try Data(contentsOf: project), saved)
        XCTAssertEqual(try store.resetCanvasDefaults(), .factory)
    }

    func testMalformedOrFuturePreferencesArePreservedByScopedReset() throws {
        let dir = try directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("app-preferences.json")
        let store = AppPreferencesStore(fileURL: file)
        let cases = [
            #"{"schemaVersion":2,"canvasDefaults":{"flowDirection":"leftToRight","nodeDetail":"medium"}}"#,
            #"{"schemaVersion":1,"canvasDefaults":{"flowDirection":"leftToRight","nodeDetail":"medium"},"unexpected":true}"#,
            #"{"schemaVersion":1,"canvasDefaults":{"flowDirection":"leftToRight","nodeDetail":"medium","unexpected":true}}"#,
            #"{"schemaVersion":1,"canvasDefaults":{"flowDirection":"diagonal","nodeDetail":"medium"}}"#,
            #"{"schemaVersion":1}"#,
            String(repeating: "x", count: 4097)
        ]
        for value in cases {
            let bytes = Data(value.utf8); try bytes.write(to: file)
            XCTAssertThrowsError(try store.load())
            XCTAssertEqual(try Data(contentsOf: file), bytes)
        }
        XCTAssertThrowsError(try store.resetCanvasDefaults())
        try store.save(.factory)
        var bad = AppPreferences.factory; bad.schemaVersion = 99
        XCTAssertThrowsError(try store.save(bad))
        XCTAssertEqual(try store.load(), .factory)
    }

    func testSymlinkAndDirectoryRejectedWithoutTouchingTarget() throws {
        let dir = try directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let target = dir.appendingPathComponent("target.json")
        try AppPreferencesStore(fileURL: target).save(.factory)
        let bytes = try Data(contentsOf: target)
        let link = dir.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        let store = AppPreferencesStore(fileURL: link)
        XCTAssertThrowsError(try store.load())
        XCTAssertThrowsError(try store.save(.factory))
        XCTAssertThrowsError(try store.resetCanvasDefaults())
        XCTAssertEqual(try Data(contentsOf: target), bytes)
        XCTAssertThrowsError(try AppPreferencesStore(fileURL: dir).load())
    }
    func testAppearanceDefaultsPersistAndLegacyMigrationDoesNotRewrite()throws {
        let dir=try directory();defer {try? FileManager.default.removeItem(at:dir)}
        let file=dir.appendingPathComponent("app-preferences.json"),store=AppPreferencesStore(fileURL:file)
        XCTAssertEqual(AppPreferences.factory.schemaVersion,3)
        XCTAssertEqual(AppPreferences.factory.appearance,.init(mode:.dark,palette:.rose,transparency:.opaque))
        let old=Data(#"{"schemaVersion":1,"canvasDefaults":{"flowDirection":"topToBottom","nodeDetail":"small"},"shortcutOverrides":{"learningLibrary":{"key":"k","command":true,"shift":true,"option":false,"control":false}}}"#.utf8)
        try old.write(to:file)
        let migrated=try store.load()
        XCTAssertEqual(migrated.schemaVersion,3)
        XCTAssertEqual(migrated.appearance,.factory)
        XCTAssertEqual(migrated.canvasDefaults,.init(flowDirection:.topToBottom,nodeDetail:.small))
        XCTAssertEqual(migrated.shortcutOverrides,["learningLibrary":ShortcutChord(key:"k",shift:true)])
        XCTAssertEqual(try Data(contentsOf:file),old,"Reading must preserve original bytes")
        var edited=migrated
        edited.appearance = .init(mode:.system,palette:.teal,transparency:.subtle)
        try store.save(edited)
        XCTAssertEqual(try AppPreferencesStore(fileURL:file).load(),edited)
        let encoded=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
        XCTAssertEqual(encoded["schemaVersion"] as? Int,3)
        XCTAssertEqual((encoded["appearance"] as? [String:Any])?["schemaVersion"] as? Int,1)
    }

    func testAppearanceAndOtherResetsAreIndependent()throws {
        let dir=try directory();defer {try? FileManager.default.removeItem(at:dir)}
        let store=AppPreferencesStore(fileURL:dir.appendingPathComponent("app-preferences.json"))
        let custom=AppPreferences(canvasDefaults:.init(flowDirection:.topToBottom,nodeDetail:.large),shortcutOverrides:["learningLibrary":ShortcutChord(key:"k",shift:true)],appearance:.init(mode:.light,palette:.teal,transparency:.subtle))
        try store.save(custom)
        let reset=try store.resetAppearance()
        XCTAssertEqual(reset.appearance,.factory)
        XCTAssertEqual(reset.canvasDefaults,custom.canvasDefaults)
        XCTAssertEqual(reset.shortcutOverrides,custom.shortcutOverrides)
        try store.save(custom)
        XCTAssertEqual(try store.resetCanvasDefaults().appearance,custom.appearance)
        try store.save(custom)
        XCTAssertEqual(try store.resetShortcuts().appearance,custom.appearance)
    }

    func testAppearanceMalformedOrFutureValuesArePreservedWithoutReset()throws {
        let dir=try directory();defer {try? FileManager.default.removeItem(at:dir)}
        let file=dir.appendingPathComponent("app-preferences.json"),store=AppPreferencesStore(fileURL:file)
        let base:[String:Any]=["schemaVersion":2,"canvasDefaults":["flowDirection":"leftToRight","nodeDetail":"medium"]]
        let valid:[String:Any]=["schemaVersion":1,"mode":"dark","palette":"rose","transparency":"opaque"]
        var invalid:[Any]=[NSNull(),[String:Any](),["schemaVersion":2,"mode":"dark","palette":"rose","transparency":"opaque"]]
        for (key,value) in [("mode","night"),("palette","rainbow"),("transparency","glass"),("extra","ignored")] {
            var valueObject=valid;valueObject[key]=value;invalid.append(valueObject)
        }
        for key in valid.keys {var value=valid;value[key]=NSNull();invalid.append(value)}
        for appearance in invalid {
            var object=base;object["appearance"]=appearance
            let bytes=try JSONSerialization.data(withJSONObject:object);try bytes.write(to:file)
            XCTAssertThrowsError(try store.load())
            XCTAssertThrowsError(try store.resetAppearance())
            XCTAssertEqual(try Data(contentsOf:file),bytes)
        }
        var legacy=base;legacy["schemaVersion"]=1;legacy["appearance"]=valid
        let bytes=try JSONSerialization.data(withJSONObject:legacy);try bytes.write(to:file)
        XCTAssertThrowsError(try store.load(),"Legacy schema cannot smuggle new settings")
        XCTAssertEqual(try Data(contentsOf:file),bytes)
        try store.save(.factory)
        var future=AppPreferences.factory;future.appearance.schemaVersion=2
        XCTAssertThrowsError(try store.save(future))
        XCTAssertEqual(try store.load(),.factory)
    }

}
