import XCTest
import KwipMLCore
@testable import KwipMLApp
final class AppShortcutActionsTests:XCTestCase {
    @MainActor func testRejectedConflictPreservesPreferencesAndRestartRestoresBinding() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let proposed=ShortcutChord(key:"k",shift:true)
        XCTAssertTrue(store.setShortcut(proposed,for:.learningLibrary))
        XCTAssertEqual(store.shortcut(for:.learningLibrary),proposed)
        let saved=try Data(contentsOf:store.appPreferenceStore.fileURL)
        XCTAssertFalse(store.setShortcut(ShortcutChord(key:"s"),for:.learningLibrary))
        XCTAssertEqual(store.shortcut(for:.learningLibrary),proposed)
        XCTAssertEqual(try Data(contentsOf:store.appPreferenceStore.fileURL),saved)
        XCTAssertTrue(store.appPreferencesError?.contains("Save project") == true)
        let reopened=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertEqual(reopened.shortcut(for:.learningLibrary),proposed)
        XCTAssertTrue(reopened.resetCanvasDefaults())
        XCTAssertEqual(reopened.shortcut(for:.learningLibrary),proposed)
        XCTAssertTrue(reopened.resetShortcuts())
        XCTAssertEqual(reopened.shortcut(for:.learningLibrary),AppShortcutRegistry.definition(for:.learningLibrary).defaultShortcut)
    }
}
