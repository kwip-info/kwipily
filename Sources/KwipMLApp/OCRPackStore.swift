import Foundation
import KwipMLCore
import KwipMLNetworking

/// Owned immutable archives. No extraction or model execution in the UI process.
enum OCRPackStore {
    static func directory(workspace:URL)->URL {workspace.appendingPathComponent("ocr-packs/"+OCRPackCatalog.id)}
    static func inputs(workspace:URL)throws->[WorkflowPreviewCapture.Input] {
        let root=directory(workspace:workspace)
        do {return try OCRPackCatalog.assets.map {asset in
            try WorkflowPreviewToolContract.verifyFile(file:.init(path:asset.name,bytes:asset.bytes,sha256:asset.sha256),directory:root,limit:asset.bytes)
            return .file(path:"input/ocr-pack/"+asset.name,source:root.appendingPathComponent(asset.name),expectedSHA256:asset.sha256)
        }} catch {throw WorkflowArchiveError("English OCR pack is missing or changed. Open Settings → Models & assets → Optional OCR model packs to download or verify the reviewed pack. No recognition was run.")}
    }
    static func installed(workspace:URL)->Bool {(try? inputs(workspace:workspace)) != nil}
    /// Conservative metadata-only reference audit. Unknown/unreadable evidence blocks removal.
    static func referenceCount(workspace:URL)throws->Int {
        var visited=0,bytes=0,count=0
        for name in ["projects","workflow-runs","workflow-operations","backend-fit-runs","backend-fit-history","workflow-histories"] {
            let root=workspace.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath:root.path) else {continue}
            let rootValues=try root.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
            guard rootValues.isDirectory==true,rootValues.isSymbolicLink != true else {throw WorkflowFileLeaseError.unsafePath}
            var enumerationFailed=false
            guard let listing=FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey],options:[],errorHandler:{_,_ in enumerationFailed=true;return false}) else {throw WorkflowArchiveError("OCR references could not be checked; pack retained.")}
            for case let url as URL in listing {
                try Task.checkCancellation();visited+=1
                guard visited<=20000 else {throw WorkflowArchiveError("OCR reference inventory exceeds its review allowance; pack retained.")}
                let values=try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
                guard values.isSymbolicLink != true else {throw WorkflowFileLeaseError.unsafePath}
                guard values.isRegularFile==true,["kwipml","json"].contains(url.pathExtension) else {continue}
                let size=values.fileSize ?? 0
                // Only graph/control records can reference an OCR operator; large result populations are not scanned.
                guard size<=2_000_000 else {throw WorkflowArchiveError("A retained metadata record exceeds its review allowance; pack retained.")}
                bytes+=size;guard bytes<=64_000_000 else {throw WorkflowArchiveError("OCR reference metadata exceeds its allowance; pack retained.")}
                let raw=try DiagnosticFileIO.read(root:url.deletingLastPathComponent(),parts:[url.lastPathComponent],limit:2_000_000)
                if String(decoding:raw,as:UTF8.self).contains(OCRConfiguration.operatorID) {count+=1}
            }
            guard !enumerationFailed else {throw WorkflowArchiveError("OCR references include unreadable directories; pack retained.")}
        }
        return count
    }
    static func removeUnreferenced(workspace:URL)throws {
        guard try referenceCount(workspace:workspace)==0 else {throw WorkflowArchiveError("Saved projects or retained History still reference OCR. Pack retained.")}
        _=try inputs(workspace:workspace)
        let held=try WorkflowFrozenSnapshotDirectory(directory:directory(workspace:workspace),removeOnDeinit:false)
        // Recheck just before removing this exact owned directory.
        guard try referenceCount(workspace:workspace)==0 else {throw WorkflowArchiveError("OCR references changed; pack retained.")}
        held.remove()
    }
    static func download(workspace:URL,accountingRoot:URL,limit:Int64?,maximumSeconds:Double,progress:@escaping @Sendable(String)->Void)async throws {
        let deadline=ProcessInfo.processInfo.systemUptime+maximumSeconds
        let temporary=workspace.appendingPathComponent("model-download-staging/"+UUID().uuidString)
        let prepare=Task.detached(priority:.utility) {()->WorkflowStorageAdmission.Reservation? in
            let reservation=try limit.map {try WorkflowStorageAdmission(workspace:workspace,accountingRoot:accountingRoot).reserve(scopes:[.init(namespace:.modelDownloadStaging,components:[temporary.lastPathComponent])],peakBytes:Int64(OCRPackCatalog.bytes),limitBytes:$0)}
            do {try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700]);return reservation} catch {reservation?.close();throw error}
        }
        let reservation=try await prepare.value;defer {reservation?.close()}
        let held=try WorkflowFrozenSnapshotDirectory(directory:temporary);defer {held.remove()}
        let scratch=try WorkflowScratchCleanup.register(directory:temporary);defer {scratch.close()}
        for asset in OCRPackCatalog.assets {
            let target=temporary.appendingPathComponent(asset.name)
            guard FileManager.default.createFile(atPath:target.path,contents:nil,attributes:[.posixPermissions:0o600]) else {throw WorkflowFileLeaseError.unsafePath}
            let output=try FileHandle(forWritingTo:target)
            do {
                var offset:Int64=0
                while offset<Int64(asset.bytes) {
                    try Task.checkCancellation();progress("Downloading English OCR · "+asset.name+" · "+String(offset/1_000_000)+" MB")
                    let request=ModelHubRequest(action:.ocrFile,repository:"JaidedAI/EasyOCR",revision:OCRPackCatalog.id,file:asset.name,offset:offset,total:Int64(asset.bytes))
                    let data=try await ModelHubBudget.read(remainingSeconds:deadline-ProcessInfo.processInfo.systemUptime){try await NetworkBrokerClient().modelHub(request)}
                    try await Task.detached(priority:.utility){try output.write(contentsOf:data)}.value
                    offset+=Int64(data.count)
                }
                try output.synchronize();try output.close()
                try await Task.detached(priority:.utility){try WorkflowPreviewToolContract.verifyFile(file:.init(path:asset.name,bytes:asset.bytes,sha256:asset.sha256),directory:temporary,limit:asset.bytes)}.value
            } catch {try? output.close();throw error}
        }
        try Task.checkCancellation()
        try await Task.detached(priority:.utility) {
            let destination=directory(workspace:workspace)
            try FileManager.default.createDirectory(at:destination.deletingLastPathComponent(),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
            // Exclusive rename. Existing installed packs are immutable, never overwritten.
            guard !FileManager.default.fileExists(atPath:destination.path) else {throw WorkflowArchiveError("English OCR already exists. Refresh its verification status.")}
            try FileManager.default.moveItem(at:temporary,to:destination)
        }.value
    }
}
