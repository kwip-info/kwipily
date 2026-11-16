import Foundation
import KwipMLCore

/// Bounded metadata inventory. Referenced and unreadable records are protected;
/// this scan never loads samples/weights or claims that an asset is unreferenced.
enum MediaLibraryInventory {
    struct Item:Identifiable,Equatable,Sendable {
        let id:String
        let projectID:String
        let kind:String
        let sha256:String
        let bytes:Int64
        let owner:String
    }
    struct Report:Sendable {
        var items:[Item]=[]
        var protectedRecords=0
        var inspectedRecords=0
        var recordsWithoutManifest=0
        var otherManifestRecords=0
        var unreadableRecords=0
        var completedRecordsWithoutMedia=0
    }
    private enum ScanFailure:Error {case limit}
    static func scan(workspace:URL,projectID:String?=nil)throws->Report {
        if let projectID,UUID(uuidString:projectID)==nil {throw WorkflowFileLeaseError.unsafePath}
        var report=Report(),visited=0,totalMetadata=0
        func directories(_ root:URL)throws->[URL] {
            guard FileManager.default.fileExists(atPath:root.path) else {return []}
            let metadata=try root.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
            guard metadata.isDirectory==true,metadata.isSymbolicLink != true else {throw WorkflowFileLeaseError.unsafePath}
            let children=try FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:[.isDirectoryKey,.isSymbolicLinkKey])
            visited+=children.count;guard visited<=10000 else {throw WorkflowArchiveError("Asset inventory exceeds its bounded scan. Assets were retained.")}
            return try children.filter {child in
                try Task.checkCancellation()
                let value=try child.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
                guard value.isSymbolicLink != true else {throw WorkflowFileLeaseError.unsafePath}
                return value.isDirectory==true && UUID(uuidString:child.lastPathComponent) != nil
            }
        }
        for namespace in ["workflow-operations","workflow-runs"] {
            for project in try directories(workspace.appendingPathComponent(namespace)) where projectID==nil || project.lastPathComponent==projectID {
                for record in try directories(project) {
                    try Task.checkCancellation();report.inspectedRecords+=1
                    guard report.inspectedRecords<=1000 else {throw WorkflowArchiveError("Asset inventory exceeds 1,000 records. Assets were retained.")}
                    let components=namespace=="workflow-operations" ? ["artifacts"]:["preparation","artifacts"]
                    let parts=[namespace,project.lastPathComponent,record.lastPathComponent]+components
                    do {
                        guard let bytes=try DiagnosticFileIO.readIfPresent(root:workspace,parts:parts+["manifest.json"],limit:2_000_000) else {report.protectedRecords+=1;report.recordsWithoutManifest+=1;continue}
                        totalMetadata+=bytes.count;guard totalMetadata<=16_000_000 else {throw ScanFailure.limit}
                        guard case .object(let manifest)=try JSONDecoder().decode(JSONValue.self,from:bytes) else {throw WorkflowFileLeaseError.unsafePath}
                        // Other operation schemas may be legitimate, but are not verified by this inventory.
                        guard manifest["schemaVersion"] == 1,manifest["kind"] == .string("dataflowExecution"),manifest["status"] == .string("completed") else {
                            report.protectedRecords+=1;report.otherManifestRecords+=1;continue
                        }
                        guard case .array(let entries)?=manifest["artifacts"],entries.count<=300,
                              entries.allSatisfy({if case .object=$0 {return true};return false}) else {throw WorkflowFileLeaseError.unsafePath}
                        var items:[Item]=[],names=Set<String>()
                        for case .object(let entry) in entries {
                            guard let kind=entry["kind"]?.stringValue,["imagePixels","audioSamples","preparedTensor"].contains(kind) else {continue}
                            guard let pin=entry["sha256"]?.stringValue,WorkflowArchiveCodec.isHash(pin),let name=entry["path"]?.stringValue,
                                  name==(kind=="imagePixels" ? "image-"+pin+".kwipimg":"media-"+pin+".kwipmedia"),names.insert(name).inserted,
                                  case .integer(let count)?=entry["bytes"],(1...50_000_000).contains(count) else {throw WorkflowFileLeaseError.unsafePath}
                            let url=(parts+[name]).reduce(workspace){$0.appendingPathComponent($1)}
                            let metadata=try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
                            guard metadata.isRegularFile==true,metadata.isSymbolicLink != true,let size=metadata.fileSize,Int64(size)==count else {throw WorkflowFileLeaseError.unsafePath}
                            items.append(.init(id:parts.joined(separator:"/")+"/"+name,projectID:project.lastPathComponent,kind:kind,sha256:pin,bytes:count,owner:namespace=="workflow-runs" ? "Recorded training History":"Owned preparation"))
                        }
                        guard report.items.count+items.count<=10000 else {throw ScanFailure.limit}
                        report.items+=items
                        if items.isEmpty {report.completedRecordsWithoutMedia+=1}
                    } catch is ScanFailure {throw WorkflowArchiveError("Asset inventory exceeds its metadata/item allowance. Assets were retained.")}
                    catch is CancellationError {throw CancellationError()}
                    catch {report.protectedRecords+=1;report.unreadableRecords+=1}
                }
            }
        }
        return report
    }
}
