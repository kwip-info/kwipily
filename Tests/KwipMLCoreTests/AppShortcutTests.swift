import XCTest
@testable import KwipMLCore

final class AppShortcutTests: XCTestCase {
    func testMissingOverridesUseSingleRegistryAndPartialOverridesPreserveOtherCommands() throws {
        let defaults = try AppShortcutRegistry.effectiveBindings()
        XCTAssertEqual(defaults.count, 10)
        XCTAssertEqual(Set(AppShortcutRegistry.commands.map(\.id)).count, 10)
        XCTAssertEqual(defaults[.newProject], ShortcutChord(key: "n"))
        XCTAssertEqual(defaults[.exportProject], ShortcutChord(key: "s", shift: true))
        XCTAssertEqual(defaults[.toggleComponents], ShortcutChord(key: "1", option: true))
        let custom = ShortcutChord(key: "k", option: true, control: true)
        let resolved = try AppShortcutRegistry.effectiveBindings(overrides: ["runFlow": custom])
        XCTAssertEqual(resolved[.runFlow], custom)
        for command in AppShortcutRegistry.commands where command.id != .runFlow {
            XCTAssertEqual(resolved[command.id], defaults[command.id])
        }
        XCTAssertEqual(try AppShortcutRegistry.effectiveBindings(overrides: [:]), defaults)
    }

    func testEffectiveCollisionsAreDeterministicAndAtomicSwapIsAllowed() throws {
        let collision = ["saveProject": ShortcutChord(key: "n")]
        XCTAssertThrowsError(try AppShortcutRegistry.effectiveBindings(overrides: collision)) {
            XCTAssertEqual($0.localizedDescription, "⌘N is assigned to both New flow and Save project.")
        }
        let swapped = try AppShortcutRegistry.effectiveBindings(overrides: [
            "saveProject": ShortcutChord(key: "n"), "newProject": ShortcutChord(key: "s")
        ])
        XCTAssertEqual(swapped[.newProject]?.key, "s")
        XCTAssertThrowsError(try AppShortcutRegistry.effectiveBindings(overrides: ["unknown": ShortcutChord(key: "k")])) {
            XCTAssertEqual($0.localizedDescription, "Unknown shortcut command: unknown.")
        }
    }

    func testAllModifierVariantsOfReservedNativeAndCanvasKeysRejected() throws {
        for key in ["z", "x", "c", "v", "a", "f", "q", "w", "h", "m", "d"] {
            for bits in 0..<8 {
                let chord = ShortcutChord(key: key, option: bits & 1 != 0, shift: bits & 2 != 0, control: bits & 4 != 0)
                XCTAssertThrowsError(try chord.validate()) {
                    XCTAssertTrue($0.localizedDescription.contains("reserved for text editing, window or canvas actions"))
                }
            }
        }
        for key in ["", "N", ",", "return", "é", "💡", " ", "nn"] {
            XCTAssertThrowsError(try ShortcutChord(key: key).validate())
        }
        XCTAssertThrowsError(try ShortcutChord(key: "k", command: false).validate())
        XCTAssertNoThrow(try ShortcutChord(key: "9", option: true, shift: true, control: true).validate())
    }

    func testStrictCodableRejectsUnknownMissingAndInvalidValues() throws {
        let chord = ShortcutChord(key: "k", option: true, shift: true, control: true)
        XCTAssertEqual(try JSONDecoder().decode(ShortcutChord.self, from: JSONEncoder().encode(chord)), chord)
        let malformed = [
            #"{"key":"k","command":true,"option":false,"shift":false,"control":false,"extra":0}"#,
            #"{"key":"k","command":true,"shift":false,"control":false}"#,
            #"{"key":"k","command":1,"option":false,"shift":false,"control":false}"#,
            #"{"key":"k","command":false,"option":false,"shift":false,"control":false}"#,
            #"{"key":"z","command":true,"option":false,"shift":false,"control":false}"#
        ]
        for json in malformed {
            XCTAssertThrowsError(try JSONDecoder().decode(ShortcutChord.self, from: Data(json.utf8)))
        }
        XCTAssertEqual(chord.displayLabel, "⌃⌥⇧⌘K")
    }
}
