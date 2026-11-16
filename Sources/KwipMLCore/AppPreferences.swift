import Foundation
import Darwin

/// Application-wide visual choices; opacity itself is a UI/accessibility policy.
public struct AppearancePreferences: Codable, Equatable {
    public enum Mode: String, Codable, CaseIterable { case dark, light, system }
    public enum Palette: String, Codable, CaseIterable { case rose, teal }
    public enum Transparency: String, Codable, CaseIterable { case opaque, subtle }
    public var schemaVersion: Int
    public var mode: Mode
    public var palette: Palette
    public var transparency: Transparency
    public static let factory = AppearancePreferences()

    public init(schemaVersion: Int = 1, mode: Mode = .dark, palette: Palette = .rose, transparency: Transparency = .opaque) {
        self.schemaVersion = schemaVersion
        self.mode = mode
        self.palette = palette
        self.transparency = transparency
    }
}

/// Application preferences only; project and named workspace records remain authoritative.
public struct AppPreferences: Codable, Equatable {
    public var schemaVersion: Int
    public var canvasDefaults: CanvasPresentationSettings
    public var shortcutOverrides: [String: ShortcutChord]?
    public var appearance: AppearancePreferences
    public var execution: ExecutionPreferences
    public var restoreCanvasSessionAtLaunch: Bool? = nil
    public var authoringRetention: FlowAuthoringRetentionPreferences? = nil
    public var canvasZoomSensitivity: Double? = nil
    public static let canvasZoomSensitivityBounds = 0.25...2.0
    public var effectiveCanvasZoomSensitivity: Double { canvasZoomSensitivity ?? 1 }
    public var runFeedback: RunFeedbackPreferences? = nil
    public var effectiveRunFeedback: RunFeedbackPreferences { runFeedback ?? .factory }
    public var effectiveAuthoringRetention: FlowAuthoringRetentionPreferences { authoringRetention ?? .factory }
    public static let factory = AppPreferences()

    public init(schemaVersion: Int = 3, canvasDefaults: CanvasPresentationSettings = .init(), shortcutOverrides: [String: ShortcutChord]? = nil, appearance: AppearancePreferences = .factory, execution: ExecutionPreferences = .factory) {
        self.schemaVersion = schemaVersion
        self.canvasDefaults = canvasDefaults
        self.shortcutOverrides = shortcutOverrides
        self.appearance = appearance
        self.execution = execution
    }

    private enum CodingKeys: String, CodingKey { case schemaVersion, canvasDefaults, shortcutOverrides, appearance, execution, restoreCanvasSessionAtLaunch, authoringRetention, runFeedback, canvasZoomSensitivity }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let version = try values.decode(Int.self, forKey: .schemaVersion)
        guard version == 1 || version == 2 || version == 3 else { throw AppPreferencesError.invalid("unsupported schema version.") }
        canvasDefaults = try values.decode(CanvasPresentationSettings.self, forKey: .canvasDefaults)
        shortcutOverrides = try values.decodeIfPresent([String: ShortcutChord].self, forKey: .shortcutOverrides)
        if version == 1 {
            guard !values.contains(.appearance) else { throw AppPreferencesError.invalid("legacy schema cannot contain appearance settings.") }
            appearance = .factory
        } else {
            appearance = try values.decode(AppearancePreferences.self, forKey: .appearance)
            guard appearance.schemaVersion == 1 else { throw AppPreferencesError.invalid("unsupported appearance schema version.") }
        }
        if version < 3 {
            guard !values.contains(.execution) else { throw AppPreferencesError.invalid("legacy schema cannot contain execution settings.") }
            execution = .factory
        } else {
            execution = try values.decode(ExecutionPreferences.self, forKey: .execution)
        }
        restoreCanvasSessionAtLaunch = try values.decodeIfPresent(Bool.self, forKey: .restoreCanvasSessionAtLaunch)
        authoringRetention = values.contains(.authoringRetention) ? try values.decode(FlowAuthoringRetentionPreferences.self, forKey: .authoringRetention) : nil
        canvasZoomSensitivity = try values.decodeIfPresent(Double.self, forKey: .canvasZoomSensitivity)
        if let speed=canvasZoomSensitivity, !speed.isFinite || !Self.canvasZoomSensitivityBounds.contains(speed) {throw AppPreferencesError.invalid("canvas zoom sensitivity must be between 0.25 and 2.")}
        runFeedback = try values.decodeIfPresent(RunFeedbackPreferences.self, forKey: .runFeedback)
        // Migration changes the returned value only; reading never rewrites disk.
        schemaVersion = 3
    }
}

public enum AppPreferencesError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let reason): return "App preferences: \(reason)" }
    }
}

/// A dedicated bounded file, never a project or arrangement directory reset.
public final class AppPreferencesStore {
    public let fileURL: URL
    private static let maximumBytes = 4096
    public init(fileURL: URL) { self.fileURL = fileURL }

    public func load() throws -> AppPreferences {
        var entry = stat()
        if lstat(fileURL.path, &entry) == 0 {
            guard (entry.st_mode & S_IFMT) == S_IFREG else {
                throw AppPreferencesError.invalid("file must be regular, not a symbolic link or directory.")
            }
        } else {
            if errno == ENOENT { return .factory }
            throw AppPreferencesError.invalid("file could not be inspected.")
        }
        let descriptor = Darwin.open(fileURL.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        if descriptor < 0 {
            throw AppPreferencesError.invalid("cannot open a regular preferences file.")
        }
        defer { Darwin.close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size >= 0, info.st_size <= Self.maximumBytes else {
            throw AppPreferencesError.invalid("file must be regular and at most 4 KiB.")
        }
        var bytes = [UInt8](repeating: 0, count: Self.maximumBytes + 1)
        var count = 0
        while count < bytes.count {
            let result = bytes.withUnsafeMutableBytes { buffer in
                Darwin.read(descriptor, buffer.baseAddress!.advanced(by: count), buffer.count - count)
            }
            if result < 0 {
                if errno == EINTR { continue }
                throw AppPreferencesError.invalid("file could not be read.")
            }
            if result == 0 { break }
            count += result
        }
        guard count <= Self.maximumBytes else { throw AppPreferencesError.invalid("file exceeds 4 KiB.") }
        return try Self.decode(Data(bytes.prefix(count)))
    }

    /// Shared pure validation for native and automation previews; never writes.
    public static func validated(_ preferences:AppPreferences)throws->AppPreferences {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return try decode(encoder.encode(preferences))
    }

    public func save(_ preferences: AppPreferences) throws {
        guard preferences.schemaVersion == 3 else { throw AppPreferencesError.invalid("unsupported schema version.") }
        var info = stat()
        if lstat(fileURL.path, &info) == 0 {
            guard (info.st_mode & S_IFMT) == S_IFREG else {
                throw AppPreferencesError.invalid("destination must be a regular file.")
            }
        } else if errno != ENOENT {
            throw AppPreferencesError.invalid("destination could not be inspected.")
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(preferences)
        _ = try Self.decode(data)
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: fileURL, options: .atomic)
    }

    @discardableResult public func resetCanvasDefaults() throws -> AppPreferences {
        var next = try load()
        next.canvasDefaults = .init()
        try save(next)
        return next
    }

    @discardableResult public func resetShortcuts() throws -> AppPreferences {
        var next = try load()
        next.shortcutOverrides = nil
        try save(next)
        return next
    }

    @discardableResult public func resetAppearance() throws -> AppPreferences {
        var next = try load()
        next.appearance = .factory
        try save(next)
        return next
    }

    @discardableResult public func resetExecution() throws -> AppPreferences {
        var next = try load()
        next.execution = try next.execution.resettingResources()
        try save(next)
        return next
    }

    private static func decode(_ data: Data) throws -> AppPreferences {
        guard data.count <= maximumBytes,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys).isSubset(of: ["schemaVersion", "canvasDefaults", "shortcutOverrides", "appearance", "execution", "restoreCanvasSessionAtLaunch", "authoringRetention", "runFeedback", "canvasZoomSensitivity"]),
              object["schemaVersion"] != nil, object["canvasDefaults"] != nil,
              let canvas = object["canvasDefaults"] as? [String: Any],
              Set(canvas.keys) == ["flowDirection", "nodeDetail"] else {
            throw AppPreferencesError.invalid("unexpected or missing fields.")
        }
        if let appearance = object["appearance"] {
            guard let fields = appearance as? [String: Any],
                  Set(fields.keys) == ["schemaVersion", "mode", "palette", "transparency"] else {
                throw AppPreferencesError.invalid("unexpected or missing appearance fields.")
            }
        }
        if let feedback = object["runFeedback"] {
            guard let fields = feedback as? [String: Any], Set(fields.keys) == ["showStatusIcon", "notifyCompleted", "notifyFailed"] else { throw AppPreferencesError.invalid("unexpected or missing run feedback fields.") }
        }
        let value = try JSONDecoder().decode(AppPreferences.self, from: data)
        try AppShortcutRegistry.validateOverrides(value.shortcutOverrides)
        return value
    }
}

public extension CanvasViewState {
    /// Presentation defaults alone must not suppress first-layout Fit.
    var isInitialViewport: Bool {
        zoom == 0.65 && scrollX == 0 && scrollY == 0 && selectedNodeIDs.isEmpty && selectedEdgeIDs.isEmpty
    }
}
