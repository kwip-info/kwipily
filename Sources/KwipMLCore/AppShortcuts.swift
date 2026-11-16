import Foundation

public enum AppCommandID: String, Codable, CaseIterable {
    case newProject, openProject, saveProject, exportProject, projectLibrary, learningLibrary
    case toggleComponents, toggleInspector, toggleResults, runFlow
}

public struct ShortcutValidationError: Error, LocalizedError, Equatable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// Canonical lowercase ASCII keys with explicit modifiers; no responder interception.
public struct ShortcutChord: Codable, Hashable {
    public var key: String
    public var command: Bool
    public var option: Bool
    public var shift: Bool
    public var control: Bool

    public init(key: String, command: Bool = true, option: Bool = false, shift: Bool = false, control: Bool = false) {
        self.key = key; self.command = command; self.option = option; self.shift = shift; self.control = control
    }

    public var displayLabel: String {
        (control ? "⌃" : "") + (option ? "⌥" : "") + (shift ? "⇧" : "") + (command ? "⌘" : "") + key.uppercased()
    }

    public func validate() throws {
        guard key.utf8.count == 1, let byte = key.utf8.first,
              (97...122).contains(byte) || (48...57).contains(byte) else {
            throw ShortcutValidationError("Choose one lowercase letter (a–z) or digit (0–9).")
        }
        guard command else { throw ShortcutValidationError("Command is required for app shortcuts.") }
        // All modifier variants are reserved: canvas select/duplicate accepts Command plus other modifiers,
        // and native responder/window commands must remain available across focus changes.
        guard !["z", "x", "c", "v", "a", "f", "q", "w", "h", "m", "d"].contains(key) else {
            throw ShortcutValidationError("\(displayLabel) is reserved for text editing, window or canvas actions.")
        }
    }

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: Key.self)
        guard Set(values.allKeys.map(\.stringValue)) == ["key", "command", "option", "shift", "control"] else {
            throw ShortcutValidationError("Shortcut has unknown or missing fields.")
        }
        key = try values.decode(String.self, forKey: Key("key"))
        command = try values.decode(Bool.self, forKey: Key("command"))
        option = try values.decode(Bool.self, forKey: Key("option"))
        shift = try values.decode(Bool.self, forKey: Key("shift"))
        control = try values.decode(Bool.self, forKey: Key("control"))
        try validate()
    }
    public func encode(to encoder: Encoder) throws {
        try validate()
        var values = encoder.container(keyedBy: Key.self)
        try values.encode(key, forKey: Key("key"))
        try values.encode(command, forKey: Key("command"))
        try values.encode(option, forKey: Key("option"))
        try values.encode(shift, forKey: Key("shift"))
        try values.encode(control, forKey: Key("control"))
    }
}

public struct AppCommandDefinition: Equatable, Identifiable {
    public let id: AppCommandID
    public let title: String
    public let defaultShortcut: ShortcutChord
}

/// The single table for settings labels, menu defaults and deterministic conflict ordering.
public enum AppShortcutRegistry {
    public static let commands: [AppCommandDefinition] = [
        .init(id: .newProject, title: "New flow", defaultShortcut: .init(key: "n")),
        .init(id: .openProject, title: "Open project…", defaultShortcut: .init(key: "o")),
        .init(id: .saveProject, title: "Save project", defaultShortcut: .init(key: "s")),
        .init(id: .exportProject, title: "Export project…", defaultShortcut: .init(key: "s", shift: true)),
        .init(id: .projectLibrary, title: "Open project library", defaultShortcut: .init(key: "p", shift: true)),
        .init(id: .learningLibrary, title: "Explore courses & examples", defaultShortcut: .init(key: "l", shift: true)),
        .init(id: .toggleComponents, title: "Toggle component ribbon", defaultShortcut: .init(key: "1", option: true)),
        .init(id: .toggleInspector, title: "Toggle configuration", defaultShortcut: .init(key: "2", option: true)),
        .init(id: .toggleResults, title: "Toggle results", defaultShortcut: .init(key: "3", option: true)),
        .init(id: .runFlow, title: "Run flow", defaultShortcut: .init(key: "r"))
    ]

    public static func definition(for command: AppCommandID) -> AppCommandDefinition {
        // Exhaustive enum/table coverage is a tested registry invariant.
        commands.first { $0.id == command }!
    }

    public static func effectiveBindings(overrides: [String: ShortcutChord]? = nil) throws -> [AppCommandID: ShortcutChord] {
        let overrides = overrides ?? [:]
        let known = Set(commands.map { $0.id.rawValue })
        if let unknown = Set(overrides.keys).subtracting(known).sorted().first {
            throw ShortcutValidationError("Unknown shortcut command: \(unknown).")
        }
        var bindings: [AppCommandID: ShortcutChord] = [:]
        var assigned: [ShortcutChord: AppCommandDefinition] = [:]
        for command in commands {
            let chord = overrides[command.id.rawValue] ?? command.defaultShortcut
            try chord.validate()
            if let existing = assigned[chord] {
                throw ShortcutValidationError("\(chord.displayLabel) is assigned to both \(existing.title) and \(command.title).")
            }
            assigned[chord] = command
            bindings[command.id] = chord
        }
        return bindings
    }

    public static func validateOverrides(_ overrides: [String: ShortcutChord]?) throws {
        _ = try effectiveBindings(overrides: overrides)
    }
}
