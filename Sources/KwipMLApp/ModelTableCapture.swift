import Foundation
import Darwin
import KwipMLCore

/// Captures complete prepared tables using bounded chunks, never a whole-file Data.
/// The role and expected pin come from the completed preparation, not a local source.
enum ModelTableCapture {
    struct Input: Sendable {
        enum Role: String, Sendable { case training, validation, evaluation, evaluationPopulation }
        let role: Role
        let url: URL
        let sha256: String
        var retainedSegments: [WorkflowArchiveFile] = []
        var path: String { "input/" + role.rawValue + ".table.json" }
    }
    static func inventory(_ inputs: [Input]) throws -> [WorkflowArchiveFile] {
        guard inputs.count == 2 && Set(inputs.map(\.role)) == [.training,.validation] || inputs.count == 3 && Set(inputs.map(\.role)) == [.training,.evaluation,.evaluationPopulation] else {
            throw WorkflowArchiveError("Connect complete Training and Validation tables from separate partitions.")
        }
        var result: [WorkflowArchiveFile] = try inputs.sorted { $0.path < $1.path }.map { input in
            guard WorkflowArchiveCodec.isHash(input.sha256) else {throw WorkflowFileLeaseError.unsafePath}
            let parent = try WorkflowEvaluationOwnership.openDirectory(input.url.deletingLastPathComponent())
            defer { Darwin.close(parent) }
            let file: WorkflowArchiveFile
            do {file = try ModelAssetFiles.stream(input.url.lastPathComponent,parent:parent,destination:nil,maximum:TableTransferLimits.artifactBytes)}
            catch is CancellationError {throw CancellationError()}
            catch {throw WorkflowArchiveError(input.role.rawValue.capitalized + " table could not be captured. Use a complete prepared table within the 50 MB transport budget; reduce unused columns or source data if needed. " + error.localizedDescription)}
            guard file.sha256 == input.sha256 else {throw WorkflowArchiveError("Prepared " + input.role.rawValue + " table changed. Run preparation again before training.")}
            return .init(path:input.path,bytes:file.bytes,sha256:file.sha256)
        }
        var segments: [String: WorkflowArchiveFile] = [:]
        for input in inputs {
            guard input.retainedSegments.count <= 512 else {throw WorkflowFileLeaseError.unsafePath}
            let parent = try WorkflowEvaluationOwnership.openDirectory(input.url.deletingLastPathComponent())
            defer {Darwin.close(parent)}
            for expected in input.retainedSegments {
                try Task.checkCancellation()
                guard PreparationArtifactPolicy.isSegmentName(expected.path),expected.bytes>0,
                      expected.bytes<=TableTransferLimits.artifactBytes,WorkflowArchiveCodec.isHash(expected.sha256) else {throw WorkflowFileLeaseError.unsafePath}
                let file = try ModelAssetFiles.stream(expected.path,parent:parent,destination:nil,maximum:TableTransferLimits.artifactBytes)
                guard file.bytes==expected.bytes,file.sha256==expected.sha256 else {throw WorkflowArchiveError("Prepared table segment changed before model capture.")}
                if let prior=segments[file.path] {guard prior==file else {throw WorkflowArchiveError("Conflicting prepared table segment identity.")}}
                else {segments[file.path]=file}
                guard segments.count+inputs.count<=512 else {throw WorkflowFileLeaseError.unsafePath}
            }
        }
        result += segments.values.sorted {$0.path<$1.path}.map {.init(path:"input/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}
        return result
    }
    static func copy(_ inputs:[Input],files:[WorkflowArchiveFile],to destination:URL)throws {
        let target = try WorkflowEvaluationOwnership.openDirectory(destination)
        defer {Darwin.close(target)}
        for input in inputs {
            try Task.checkCancellation()
            guard let expected=files.first(where:{$0.path==input.path}) else {throw WorkflowFileLeaseError.unsafePath}
            let parent=try WorkflowEvaluationOwnership.openDirectory(input.url.deletingLastPathComponent())
            defer {Darwin.close(parent)}
            let file=try ModelAssetFiles.stream(input.url.lastPathComponent,parent:parent,destination:target,maximum:TableTransferLimits.artifactBytes,destinationName:input.role.rawValue+".table.json")
            guard file.bytes==expected.bytes,file.sha256==expected.sha256 else {throw WorkflowArchiveError("Prepared " + input.role.rawValue + " table changed during capture. Run preparation again.")}
        }
        var copied=Set<String>()
        for input in inputs {
            let parent=try WorkflowEvaluationOwnership.openDirectory(input.url.deletingLastPathComponent())
            defer {Darwin.close(parent)}
            for segment in input.retainedSegments {
                try Task.checkCancellation()
                guard PreparationArtifactPolicy.isSegmentName(segment.path),let expected=files.first(where:{$0.path=="input/"+segment.path}),
                      expected.bytes==segment.bytes,expected.sha256==segment.sha256 else {throw WorkflowFileLeaseError.unsafePath}
                let file=try ModelAssetFiles.stream(segment.path,parent:parent,destination:copied.contains(segment.path) ? nil:target,maximum:TableTransferLimits.artifactBytes)
                guard file.bytes==expected.bytes,file.sha256==expected.sha256 else {throw WorkflowArchiveError("Prepared table segment changed during capture.")}
                copied.insert(segment.path)
            }
        }
    }
}
