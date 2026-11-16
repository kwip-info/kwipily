import Foundation

public enum RuntimeLocationError: LocalizedError {
    case invalidBundle(String)
    case unavailable
    public var errorDescription: String? {
        switch self {
        case .invalidBundle(let reason): return "Bundled runtime is invalid: \(reason). Reinstall the application; development fallback is disabled for a present bundle."
        case .unavailable: return "No bundled runtime is available. Development runtime requires explicit opt-in and a configured checkout."
        }
    }
}

/// Runtime code is read-only bundle content. User documents/artifacts belong elsewhere.
public struct RuntimeLocation: Equatable {
    public enum Kind: String { case bundled, development }
    public let kind: Kind
    public let packageVersions: [String: String]
    public let executableURL: URL
    public let entryURL: URL
    public let registryURL: URL
    public let workingDirectoryURL: URL
    public var argumentPrefix: [String] { (kind == .bundled ? ["-I", "-B"] : []) + [entryURL.path] }

    public enum BrokerEntry: String, CaseIterable {
        case databaseCapture = "database_source_capture.py"
        case databaseOutput = "database_output_service.py"
        case apiTable = "api_table_service.py"
    }
    /// Fixed broker programs belong to the verified runtime, never a caller path.
    public func brokerEntry(_ entry: BrokerEntry) throws -> URL {
        let directory = kind == .bundled ? workingDirectoryURL.appendingPathComponent("runner", isDirectory: true) : entryURL.deletingLastPathComponent()
        let directoryValues = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        let script = directory.appendingPathComponent(entry.rawValue)
        let values = try script.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard directoryValues.isDirectory == true, directoryValues.isSymbolicLink != true,
              values.isRegularFile == true, values.isSymbolicLink != true,
              script.resolvingSymlinksInPath().path.hasPrefix(workingDirectoryURL.resolvingSymlinksInPath().path + "/"),
              script.resolvingSymlinksInPath().deletingLastPathComponent() == directory.resolvingSymlinksInPath() else {
            throw RuntimeLocationError.invalidBundle("missing or escaping broker entry")
        }
        return script
    }

    private struct Manifest: Decodable {
        let schemaVersion: Int
        let pythonExecutable: String
        let entryPoint: String
        let registry: String
        let pythonVersion: String
        let packages: [String: String]
    }
    /// One physical runtime lives inside the database helper so its sandbox can
    /// read executable code. Older applications retain their main-resource path.
    public static func bundledRuntimeRoot(resources: URL?) throws -> URL? {
        guard let resources else { return nil }
        let legacy = resources.appendingPathComponent("KwipMLRuntime", isDirectory: true)
        let nested = resources.deletingLastPathComponent().appendingPathComponent("XPCServices/DatabaseSourceService.xpc/Contents/Resources/KwipMLRuntime", isDirectory: true)
        let manager = FileManager.default
        func present(_ url: URL) -> Bool {
            manager.fileExists(atPath: url.path) || (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
        }
        guard !(present(legacy) && present(nested)) else { throw RuntimeLocationError.invalidBundle("multiple runtime payloads") }
        let root = present(nested) ? nested : present(legacy) ? legacy : nil
        if let root {
            let values = try root.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
            guard values.isSymbolicLink != true, values.isDirectory == true else { throw RuntimeLocationError.invalidBundle("runtime must be a physical directory") }
            let physicalContents = resources.deletingLastPathComponent().resolvingSymlinksInPath().path + "/"
            guard root.resolvingSymlinksInPath().path.hasPrefix(physicalContents) else { throw RuntimeLocationError.invalidBundle("runtime escapes application contents") }
        }
        return root
    }
    public static func resolve(resources: URL?, developmentRoot: URL? = nil, allowDevelopment: Bool = false) throws -> RuntimeLocation {
        let manager = FileManager.default
        if let root = try bundledRuntimeRoot(resources: resources) {
                do {
                    let manifestURL = root.appendingPathComponent("runtime-manifest.json")
                    guard let size = try manifestURL.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 65536 else { throw RuntimeLocationError.invalidBundle("manifest exceeds 64 KB") }
                    let data = try Data(contentsOf: manifestURL)
                    guard data.count <= 65536 else { throw RuntimeLocationError.invalidBundle("manifest exceeds 64 KB") }
                    let manifest = try JSONDecoder().decode(Manifest.self, from: data)
                    guard manifest.schemaVersion == 1, !manifest.pythonVersion.isEmpty, !(manifest.packages["torch"] ?? "").isEmpty else { throw RuntimeLocationError.invalidBundle("unsupported or incomplete manifest") }
                    func resolvePath(_ relative: String) throws -> URL {
                        guard !relative.isEmpty, !relative.hasPrefix("/"), !relative.split(separator: "/").contains("..") else { throw RuntimeLocationError.invalidBundle("path escapes runtime") }
                        let url = root.appendingPathComponent(relative).standardizedFileURL
                        let physicalRoot = root.resolvingSymlinksInPath().path + "/"
                        guard url.resolvingSymlinksInPath().path.hasPrefix(physicalRoot), manager.fileExists(atPath: url.path),
                              try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { throw RuntimeLocationError.invalidBundle("missing or escaping runtime file") }
                        return url
                    }
                    let executable = try resolvePath(manifest.pythonExecutable)
                    guard manager.isExecutableFile(atPath: executable.path) else { throw RuntimeLocationError.invalidBundle("Python is not executable") }
                    return RuntimeLocation(kind: .bundled, packageVersions: manifest.packages, executableURL: executable, entryURL: try resolvePath(manifest.entryPoint), registryURL: try resolvePath(manifest.registry), workingDirectoryURL: root)
                } catch let error as RuntimeLocationError { throw error }
                catch { throw RuntimeLocationError.invalidBundle(error.localizedDescription) }
        }
        guard allowDevelopment, let root = developmentRoot else { throw RuntimeLocationError.unavailable }
        let python = root.appendingPathComponent(".venv/bin/python")
        let entry = root.appendingPathComponent("runner/main.py")
        let registry = root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json")
        guard manager.isExecutableFile(atPath: python.path), manager.fileExists(atPath: entry.path), manager.fileExists(atPath: registry.path) else { throw RuntimeLocationError.unavailable }
        return RuntimeLocation(kind: .development, packageVersions: [:], executableURL: python, entryURL: entry, registryURL: registry, workingDirectoryURL: root)
    }
}
