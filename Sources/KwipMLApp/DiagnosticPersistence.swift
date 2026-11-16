import Foundation
import CryptoKit
import KwipMLCore

struct DiagnosticArchiveReference:Codable,Equatable {var schemaVersion=1;var sha256:String}
enum DiagnosticArchiveLoad {case legacy,verified(DiagnosticReport),unavailable(String)}
private struct DiagnosticStoredObservation:Codable {
    var epoch:Int
    var trainingBits:String
    var validationBits:String
    var accuracyBits:String?
    init(_ metric:EpochMetric) {epoch=metric.epoch;trainingBits=String(metric.trainLoss.bitPattern,radix:16);validationBits=String(metric.validationLoss.bitPattern,radix:16);accuracyBits=metric.accuracy.map {String($0.bitPattern,radix:16)}}
    func metric()throws->EpochMetric {
        func value(_ text:String)throws->Double {
            guard let bits=UInt64(text,radix:16),String(bits,radix:16)==text else {throw DiagnosticContractError("Invalid diagnostic observation bit pattern.")}
            return Double(bitPattern:bits)
        }
        return try EpochMetric(epoch:epoch,trainLoss:value(trainingBits),validationLoss:value(validationBits),accuracy:accuracyBits.map {try value($0)})
    }
}
private struct DiagnosticArchive:Codable {
    var schemaVersion=1
    var projectID:String
    var experimentID:String
    var status:String
    var recordSHA256:String
    var manifestSHA256:String?
    var observations:[DiagnosticStoredObservation]
    var rawError:String?
    var registry:DiagnosticRuleRegistry
    var report:DiagnosticReport
}

enum DiagnosticPersistence {
    static func digest(_ bytes:Data)->String {SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined()}
    private static func encode<T:Encodable>(_ value:T)throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return try encoder.encode(value)}
    private static func recordDigest(_ record:ExperimentRecord)throws->String {var copy=record;copy.diagnosticReport=nil;copy.diagnosticError=nil;return digest(try encode(copy))}
    private static func parts(projectID:String,experimentID:String)throws->[String] {
        guard UUID(uuidString:projectID) != nil,UUID(uuidString:experimentID) != nil else {throw DiagnosticContractError("Invalid diagnostic project or experiment identity.")}
        return ["diagnostic-reports",projectID,experimentID+".json"]
    }
    static func completedManifest(record:ExperimentRecord,projectID:String,root:URL)throws->Data {
        let raw=record.directory.split(separator:"/",omittingEmptySubsequences:false)
        guard record.directory.hasPrefix("/"), !raw.contains(".."), !raw.contains("."), !record.directory.contains("\0") else {
            throw DiagnosticContractError("Completed diagnostics require an app-owned run directory.")
        }
        let base=root.standardizedFileURL.pathComponents
        let path=URL(fileURLWithPath:record.directory).standardizedFileURL.pathComponents
        guard path.starts(with:base) else {throw DiagnosticContractError("Completed diagnostics require an app-owned run directory.")}
        let relative=Array(path.dropFirst(base.count))
        let legacy=relative.count==3 && relative[0]=="runs" && UUID(uuidString:relative[1]) != nil && relative[2]=="artifacts"
        let operation=relative.count==4 && relative[0]=="workflow-operations" && relative[1]==projectID
            && UUID(uuidString:projectID)?.uuidString==projectID && UUID(uuidString:relative[2])?.uuidString==relative[2] && relative[3]=="artifacts"
        guard legacy || operation else {throw DiagnosticContractError("Completed diagnostics require an app-owned run directory in the expected project.")}
        let data=try DiagnosticFileIO.read(root:root,parts:relative+["manifest.json"])
        struct Header:Decodable {var schemaVersion:Int;var status:String;var graph:GraphDocument}
        let manifest=try JSONDecoder().decode(Header.self,from:data)
        guard manifest.schemaVersion==1,manifest.status=="completed",manifest.graph==record.graph else {throw DiagnosticContractError("Completed manifest schema, status or graph differs from the experiment.")}
        return data
    }
    static func save(record:ExperimentRecord,projectID:String,metrics:[EpochMetric],root:URL)throws->DiagnosticArchiveReference {
        guard ["completed","failed","cancelled"].contains(record.status),metrics.count<=100_000 else {throw DiagnosticContractError("Diagnostic snapshot exceeds bounds or has a nonterminal status.")}
        let path=try parts(projectID:projectID,experimentID:record.id)
        let manifest=record.status=="completed" ? try digest(completedManifest(record:record,projectID:projectID,root:root)):nil
        let epochs=metrics.map(\.epoch)
        let sampled=(epochs.first ?? 1)>1 || zip(epochs,epochs.dropFirst()).contains {$1>$0 && ($1.subtractingReportingOverflow($0).overflow || $1.subtractingReportingOverflow($0).partialValue>1)}
        let context=DiagnosticInputContext(runID:record.id,sampling:sampled ? .sampled:.unspecified)
        let registry=DiagnosticRuleRegistry.builtIn
        let report=try RunDiagnostics.report(metrics,error:record.error,context:context,registry:registry)
        let archive=DiagnosticArchive(projectID:projectID,experimentID:record.id,status:record.status,recordSHA256:try recordDigest(record),manifestSHA256:manifest,observations:metrics.map(DiagnosticStoredObservation.init),rawError:record.error,registry:registry,report:report)
        let data=try encode(archive)
        try DiagnosticFileIO.writeImmutable(data,root:root,parts:path)
        return DiagnosticArchiveReference(sha256:digest(data))
    }
    static func load(record:ExperimentRecord,projectID:String,root:URL)->DiagnosticArchiveLoad {
        if let error=record.diagnosticError {return .unavailable(error)}
        guard let reference=record.diagnosticReport else {return .legacy}
        do {
            guard reference.schemaVersion==1,reference.sha256.count==64 else {throw DiagnosticContractError("Unsupported diagnostic artifact reference.")}
            let data=try DiagnosticFileIO.read(root:root,parts:parts(projectID:projectID,experimentID:record.id))
            guard digest(data)==reference.sha256 else {throw DiagnosticContractError("Saved diagnostic artifact digest differs from its experiment pin.")}
            let object=try DiagnosticSchema.object(data)
            try DiagnosticSchema.keys(object,["schemaVersion","projectID","experimentID","status","recordSHA256","manifestSHA256","observations","rawError","registry","report"])
            guard let rawRegistry=object["registry"],let rawReport=object["report"] else {throw DiagnosticContractError("Missing pinned diagnostic registry or report.")}
            _ = try DiagnosticRuleRegistry.decode(JSONSerialization.data(withJSONObject:rawRegistry))
            _ = try DiagnosticReport.decode(JSONSerialization.data(withJSONObject:rawReport))
            for row in object["observations"] as? [[String:Any]] ?? [] {try DiagnosticSchema.keys(row,["epoch","trainingBits","validationBits","accuracyBits"])}
            let archive=try JSONDecoder().decode(DiagnosticArchive.self,from:data)
            guard archive.schemaVersion==1,archive.projectID==projectID,archive.experimentID==record.id,
                  archive.status==record.status,["completed","failed","cancelled"].contains(archive.status),
                  archive.recordSHA256 == (try recordDigest(record)),archive.rawError==record.error,archive.observations.count<=100_000,
                  archive.report.context.runID==record.id else {throw DiagnosticContractError("Saved diagnostic artifact differs from the terminal experiment identity.")}
            if record.status=="completed" {
                guard archive.manifestSHA256 == (try digest(completedManifest(record:record,projectID:projectID,root:root))) else {throw DiagnosticContractError("Completed manifest no longer matches the recorded diagnostic pin.")}
            } else if archive.manifestSHA256 != nil {throw DiagnosticContractError("Interrupted diagnostic snapshot cannot claim a completed manifest.")}
            let metrics=try archive.observations.map {try $0.metric()}
            let replay=try RunDiagnostics.report(metrics,error:archive.rawError,context:archive.report.context,registry:archive.registry)
            guard replay==archive.report else {throw DiagnosticContractError("Saved diagnostic report does not reproduce from its original observations and pinned rules.")}
            return .verified(archive.report)
        } catch {return .unavailable(error.localizedDescription)}
    }
}

extension WorkspaceStore {
    func loadExperimentDiagnostics(_ record:ExperimentRecord)->DiagnosticArchiveLoad {
        guard let projectID=activeProjectID else {return .unavailable("Open the experiment's project before loading its diagnostics.")}
        return DiagnosticPersistence.load(record:record,projectID:projectID,root:workspaceDirectory)
    }
}
