import Foundation
import Combine
import KwipMLCore

struct MeasuredDiagnosticSuppression:Codable,Equatable,Identifiable {
    let id:UUID
    let runID:String
    let inputSHA256:String
    let findingID:String
    let reason:String
    let createdAt:Date
    let expiresAt:Date
    init(report:MeasuredDiagnosticReport,finding:MeasuredDiagnosticFinding,reason:String,now:Date,expiresAt:Date)throws {
        try report.validate()
        let trimmed=reason.trimmingCharacters(in:.whitespacesAndNewlines)
        guard report.findings.contains(finding),!trimmed.isEmpty,trimmed.count<=512,now.timeIntervalSince1970.isFinite,
              expiresAt.timeIntervalSince1970.isFinite,expiresAt>now,expiresAt.timeIntervalSince(now)<=30*86400 else {throw DiagnosticContractError("Suppression needs a reason and an expiry within thirty days.")}
        id=UUID();runID=report.runID;inputSHA256=report.inputSHA256;findingID=finding.id;self.reason=trimmed;createdAt=now;self.expiresAt=expiresAt
    }
    func applies(to report:MeasuredDiagnosticReport,finding:MeasuredDiagnosticFinding,at date:Date)->Bool {
        runID==report.runID && inputSHA256==report.inputSHA256 && findingID==finding.id && date>=createdAt && date<expiresAt
    }
    func validate()throws {
        guard UUID(uuidString:runID) != nil,WorkflowArchiveCodec.isHash(inputSHA256),!findingID.isEmpty,findingID.count<=512,
              !reason.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,reason.count<=512,
              createdAt.timeIntervalSince1970.isFinite,expiresAt.timeIntervalSince1970.isFinite,expiresAt>createdAt,expiresAt.timeIntervalSince(createdAt)<=30*86400 else {throw DiagnosticContractError("Invalid diagnostic suppression.")}
    }
}
@MainActor final class MeasuredDiagnosticSuppressionStore:ObservableObject {
    @Published private(set) var entries:[MeasuredDiagnosticSuppression]=[]
    @Published private(set) var error:String?
    private let root:URL
    private let projectID:String
    init(root:URL,projectID:String) {self.root=root;self.projectID=projectID;reload()}
    private var parts:[String] {["diagnostic-workbench",projectID,"suppressions.json"]}
    func reload() {
        do {
            guard UUID(uuidString:projectID) != nil else {throw DiagnosticContractError("Invalid project for diagnostic annotations.")}
            guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:1_000_000) else {entries=[];error=nil;return}
            let decoded=try JSONDecoder().decode([MeasuredDiagnosticSuppression].self,from:bytes)
            guard decoded.count<=1000,Set(decoded.map(\.id)).count==decoded.count else {throw DiagnosticContractError("Too many or duplicate diagnostic suppressions.")}
            try decoded.forEach{try $0.validate()};entries=decoded;error=nil
        } catch {self.error=error.localizedDescription}
    }
    func save(_ value:MeasuredDiagnosticSuppression,now:Date=Date())throws {
        try value.validate()
        // All mutations are MainActor synchronous: reload + replace is one
        // non-interleaved operation across multiple History views in this app.
        reload()
        guard error==nil else {throw DiagnosticContractError(error!)}
        var next=entries.filter{$0.expiresAt>now && !($0.runID==value.runID && $0.findingID==value.findingID)}
        next.append(value);guard next.count<=1000 else {throw DiagnosticContractError("Diagnostic suppression limit reached.")}
        try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(next),root:root,parts:parts);entries=next
    }
    func remove(_ id:UUID)throws {
        reload()
        guard error==nil else {throw DiagnosticContractError(error!)}
        let next=entries.filter{$0.id != id};try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(next),root:root,parts:parts);entries=next
    }
}
/// Sharing is allowlisted. No raw error, local node ID, evidence path, source row,
/// activation value, suppression reason or captured configuration is copied.
enum MeasuredDiagnosticExport {
    static func redacted(_ report:MeasuredDiagnosticReport)throws->Data {
        try report.validate()
        struct Summary:Encodable {let ruleID:String;let title:String;let occurrences:Int;let prerequisites:[String];let confounders:[String];let proposedExperiment:String}
        struct Export:Encodable {let schemaVersion:Int;let disclosure:String;let registryID:String;let registryVersion:Int;let findings:[Summary]}
        let items=MeasuredDiagnostics.rules.compactMap {rule->Summary? in
            let count=report.findings.filter{$0.ruleID==rule.id}.count
            return count>0 ? .init(ruleID:rule.id,title:rule.title,occurrences:count,prerequisites:rule.prerequisites,confounders:rule.confounders,proposedExperiment:rule.experiment):nil
        }
        return try WorkflowArchiveCodec.encode(Export(schemaVersion:1,disclosure:"redacted diagnostic summary; sensitive samples, identifiers, raw errors, private annotations and paths omitted",registryID:report.registryID,registryVersion:report.registryVersion,findings:items))
    }
}
