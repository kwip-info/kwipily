import Foundation
import Darwin
import KwipMLCore

/// Private source inputs for one universal run. Admission precedes every payload copy.
final class WorkflowRunSourceCapture {
    let sources: [WorkflowNodeAddress: WorkflowCapturedSource]
    private let lifetime: WorkflowExportStorageLifetime?
    private let collections:[FileInputCollectionCapture]
    private let retainedRegular:WorkflowRunSourceCapture?

    private init(sources: [WorkflowNodeAddress: WorkflowCapturedSource], lifetime: WorkflowExportStorageLifetime?,collections:[FileInputCollectionCapture]=[],retainedRegular:WorkflowRunSourceCapture?=nil) {
        self.sources = sources; self.lifetime = lifetime;self.collections=collections;self.retainedRegular=retainedRegular
    }
    func remove() { lifetime?.remove();collections.forEach {$0.remove()};retainedRegular?.remove() }
    deinit { remove() }

    static func capture(paths: [WorkflowNodeAddress: String], workspace: URL, accountingRoot: URL? = nil, sourcePermissions: URL,
                        storageLimitBytes: Int64?,fileConfigurations:[WorkflowNodeAddress:[String:JSONValue]]=[:],progress:@escaping @Sendable (WorkflowSourceCaptureProgress)->Void = {_ in}) async throws -> WorkflowRunSourceCapture {
        try Task.checkCancellation()
        let worker = Task.detached(priority: .userInitiated) {
            try captureNow(paths: paths, workspace: workspace, accountingRoot: accountingRoot, sourcePermissions: sourcePermissions,
                           storageLimitBytes: storageLimitBytes,fileConfigurations:fileConfigurations,progress:progress)
        }
        let captured = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
        guard !Task.isCancelled else { captured.remove(); throw CancellationError() }
        return captured
    }

    private static func captureNow(paths:[WorkflowNodeAddress:String],workspace:URL,accountingRoot:URL?,sourcePermissions:URL,storageLimitBytes:Int64?,fileConfigurations:[WorkflowNodeAddress:[String:JSONValue]],progress:@Sendable (WorkflowSourceCaptureProgress)->Void)throws->WorkflowRunSourceCapture {
        guard paths.count<=128,paths.keys.allSatisfy({$0.domain == .data}),paths.values.allSatisfy({!$0.isEmpty && !$0.contains("\0") && $0.utf8.count<=4096}),Set(fileConfigurations.keys).isSubset(of:Set(paths.keys)) else {throw WorkflowFileLeaseError.unsafePath}
        if fileConfigurations.isEmpty {return try captureRegularNow(paths:paths,workspace:workspace,accountingRoot:accountingRoot,sourcePermissions:sourcePermissions,storageLimitBytes:storageLimitBytes,progress:progress)}
        var collections:[FileInputCollectionCapture]=[],retained=false
        defer{if !retained{collections.forEach {$0.remove()}}}
        var sources:[WorkflowNodeAddress:WorkflowCapturedSource]=[:]
        for (index,address) in fileConfigurations.keys.sorted(by:{$0.localID<$1.localID}).enumerated() {
            try Task.checkCancellation();progress(.init(address:address,sequence:index*2+1,finished:false))
            let capture=try FileInputCollectionCapture.capture(path:paths[address]!,configuration:fileConfigurations[address]!,workspace:workspace,accountingRoot:accountingRoot,sourcePermissions:sourcePermissions,storageLimitBytes:storageLimitBytes)
            collections.append(capture);sources[address]=capture.source;progress(.init(address:address,sequence:index*2+2,finished:true))
        }
        let regular=try captureRegularNow(paths:paths.filter {fileConfigurations[$0.key]==nil},workspace:workspace,accountingRoot:accountingRoot,sourcePermissions:sourcePermissions,storageLimitBytes:storageLimitBytes,progress:{progress(.init(address:$0.address,sequence:$0.sequence+fileConfigurations.count*2,finished:$0.finished))})
        for (address,source) in regular.sources {sources[address]=source}
        guard sources.values.reduce(0,{$0+$1.totalCapturedByteCount})<=268_435_456 else {regular.remove();throw WorkflowArchiveError("Run source collections exceed the aggregate capture budget.")}
        retained=true;return .init(sources:sources,lifetime:nil,collections:collections,retainedRegular:regular)
    }
    private static func captureRegularNow(paths: [WorkflowNodeAddress: String], workspace: URL, accountingRoot: URL? = nil, sourcePermissions: URL,
                                   storageLimitBytes: Int64?,progress:@Sendable (WorkflowSourceCaptureProgress)->Void) throws -> WorkflowRunSourceCapture {
        try Task.checkCancellation()
        guard paths.count <= 128, paths.keys.allSatisfy({ $0.domain == .data }),
              paths.values.allSatisfy({ !$0.isEmpty && !$0.contains("\0") && $0.count <= 4096 }),
              storageLimitBytes == nil || storageLimitBytes! >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        guard !paths.isEmpty else { return .init(sources: [:], lifetime: nil) }
        let access = SourceAccess(directory: sourcePermissions)
        let addresses = paths.keys.sorted { $0.localID < $1.localID }
        var bounds: [WorkflowNodeAddress: Int] = [:], total = 0
        for address in addresses {
            try Task.checkCancellation()
            let count = try access.withAccess(to: paths[address]!) { url in
                try WorkflowPreviewCapture.byteCountBound(inputs: [
                    .file(path: "input/data/" + url.lastPathComponent, source: url, expectedSHA256: nil)
                ], purpose: .sourceInspection)
            }
            guard count <= 256_000_000 - total else { throw WorkflowArchiveError("Run sources exceed the 256 MB capture allowance.") }
            bounds[address] = count; total += count
        }
        try Task.checkCancellation()
        let stageID = UUID()
        // A nonempty set of zero-byte sources needs a live positive ledger record.
        let reservation = try storageLimitBytes.map { limit in
            try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot).reserve(scopes: [
                .init(namespace: .workflowSourceStaging, components: [stageID.uuidString])
            ], peakBytes: Int64(max(1, total)), limitBytes: limit)
        }
        var lifetime: WorkflowExportStorageLifetime?, retained = false
        defer {
            if !retained {
                if let lifetime { lifetime.remove() } else { reservation?.close() }
            }
        }
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        guard let resolved = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let root = URL(fileURLWithPath: String(cString: resolved)); free(resolved)
        let rootFD = try WorkflowEvaluationOwnership.openDirectory(root)
        defer { Darwin.close(rootFD) }
        let rootIdentity = try WorkflowEvaluationOwnership.directoryIdentity(rootFD)
        let namespace = "workflow-source-staging"
        if mkdirat(rootFD, namespace, 0o700) != 0, errno != EEXIST { throw WorkflowFileLeaseError.system(errno) }
        let parentFD = openat(rootFD, namespace, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parentFD >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(parentFD) }
        let parentIdentity = try WorkflowEvaluationOwnership.directoryIdentity(parentFD)
        let parent = root.appendingPathComponent(namespace)
        func verify(_ url: URL, _ identity: WorkflowFileIdentity) throws {
            let fd = try WorkflowEvaluationOwnership.openDirectory(url)
            defer { Darwin.close(fd) }
            guard try WorkflowEvaluationOwnership.directoryIdentity(fd) == identity else { throw WorkflowFileLeaseError.unsafePath }
        }
        try verify(root, rootIdentity); try verify(parent, parentIdentity)
        guard mkdirat(parentFD, stageID.uuidString, 0o700) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        let directory = parent.appendingPathComponent(stageID.uuidString)
        let stageFD = openat(parentFD, stageID.uuidString, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard stageFD >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(stageFD) }
        let stageIdentity = try WorkflowEvaluationOwnership.directoryIdentity(stageFD)
        try verify(directory, stageIdentity)
        lifetime = try WorkflowExportStorageLifetime(directory: directory, reservation: reservation)
        var sources: [WorkflowNodeAddress: WorkflowCapturedSource] = [:]
        for (index, address) in addresses.enumerated() {
            try Task.checkCancellation()
            progress(.init(address:address,sequence:index*2+1,finished:false))
            let bound = bounds[address]!
            let ext = URL(fileURLWithPath: paths[address]!).pathExtension.lowercased()
            let safeExtension = !ext.isEmpty && ext.count <= 10 && ext.utf8.allSatisfy { (97...122).contains($0) || (48...57).contains($0) } ? ext : "data"
            let name = "source-\(index).\(safeExtension)"
            let receipt = try access.withAccess(to: paths[address]!) { url in
                try DiagnosticFileIO.copySelectedFile(url, root: directory, parts: [name], limit: bound)
            }
            try Task.checkCancellation()
            try verify(root, rootIdentity); try verify(parent, parentIdentity); try verify(directory, stageIdentity)
            sources[address] = .init(url: directory.appendingPathComponent(name), sha256: receipt.sha256, byteCount: receipt.bytes)
            progress(.init(address:address,sequence:index*2+2,finished:true))
        }
        try Task.checkCancellation()
        try verify(root, rootIdentity); try verify(parent, parentIdentity); try verify(directory, stageIdentity)
        retained = true
        return .init(sources: sources, lifetime: lifetime)
    }
}
