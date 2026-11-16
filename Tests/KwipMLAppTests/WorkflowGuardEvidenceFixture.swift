import Foundation
import KwipMLCore
@testable import KwipMLApp

/// Provenance-only fixture for mocked launchers. These bytes are never loaded as code.
enum WorkflowGuardEvidenceFixture {
    static func stage(_ runtime:RuntimeLocation)throws {
        let root=runtime.workingDirectoryURL
        let native=root.appendingPathComponent("runner/native")
        let staged=root.appendingPathComponent(".local/native-watchdog")
        try FileManager.default.createDirectory(at:native,withIntermediateDirectories:true)
        try FileManager.default.createDirectory(at:staged,withIntermediateDirectories:true)
        let source=Data("mock lifetime guard source".utf8),library=Data("mock lifetime guard library; never executed".utf8)
        try source.write(to:native.appendingPathComponent("lifetime_watchdog.c"))
        try library.write(to:staged.appendingPathComponent("lifetime_watchdog.dylib"))
        let manifest:[String:Any] = ["schemaVersion":1,"sourceSHA256":WorkflowPhaseArtifacts.hash(source),"librarySHA256":WorkflowPhaseArtifacts.hash(library)]
        try JSONSerialization.data(withJSONObject:manifest,options:[.sortedKeys]).write(to:staged.appendingPathComponent("lifetime_watchdog.json"))
    }
}
