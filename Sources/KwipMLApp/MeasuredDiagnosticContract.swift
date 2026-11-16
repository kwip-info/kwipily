import Foundation
import KwipMLCore

/// Additional measured checks have their own identity. The original seven-rule
/// registry and archived reports retain their exact historical interpretation.
struct MeasuredMetric:Codable,Equatable {
    let epoch:Int
    let trainBits:String
    let validationBits:String
    let accuracyBits:String?
    init(_ metric:EpochMetric) {
        epoch=metric.epoch;trainBits=String(metric.trainLoss.bitPattern,radix:16)
        validationBits=String(metric.validationLoss.bitPattern,radix:16);accuracyBits=metric.accuracy.map {String($0.bitPattern,radix:16)}
    }
    var metric:EpochMetric? {
        guard let train=UInt64(trainBits,radix:16),let validation=UInt64(validationBits,radix:16) else {return nil}
        var accuracy:Double?
        if let accuracyBits {guard let bits=UInt64(accuracyBits,radix:16) else {return nil};accuracy=Double(bitPattern:bits)}
        return .init(epoch:epoch,trainLoss:Double(bitPattern:train),validationLoss:Double(bitPattern:validation),accuracy:accuracy)
    }
}
struct MeasuredGradientBatch:Codable,Equatable {
    let epoch:Int
    let batch:Int
    let presentElements:Int
    let nonfiniteElements:Int
    let zeroElements:Int
    let missingParameters:Int
    let l2Norm:Double?
    var reference:String {"gradient-summary.json:epoch=\(epoch),batch=\(batch)"}
}
struct MeasuredActivationSample:Codable,Equatable {
    let nodeID:String
    let component:String
    let values:[Double]
}
struct MeasuredDiagnosticInput:Codable,Equatable {
    var schemaVersion=1
    var runID:String
    var requestSHA256:String
    var workflowSHA256:String?
    var capturedTrain:WorkflowNodeAddress?
    var backend="pytorch"
    var metrics:[MeasuredMetric]
    var continuation=false
    var gradients:[MeasuredGradientBatch]=[]
    var activations:[MeasuredActivationSample]=[]
    var validationRows:Int?
    var rawError:String?
    init(runID:String,requestSHA256:String,metrics:[EpochMetric]) {
        self.runID=runID;self.requestSHA256=requestSHA256;self.metrics=metrics.map(MeasuredMetric.init)
    }
    func validate()throws {
        guard schemaVersion==1,UUID(uuidString:runID) != nil,WorkflowArchiveCodec.isHash(requestSHA256),
              workflowSHA256.map(WorkflowArchiveCodec.isHash) ?? true,
              capturedTrain.map({$0.domain == .control && !$0.localID.isEmpty && $0.localID.count<=128}) ?? true,
              ["pytorch","tensorflow","sklearn"].contains(backend),metrics.count<=100000,metrics.allSatisfy({$0.metric != nil}),
              gradients.count<=1024,activations.count<=128,rawError.map({$0.utf8.count<=1_000_000}) ?? true,
              validationRows.map({(1...100000).contains($0)}) ?? true else {throw DiagnosticContractError("Invalid measured diagnostic input identity or bounds.")}
        guard Set(gradients.map(\.reference)).count==gradients.count,
              gradients.allSatisfy({b in
                (1...10000).contains(b.epoch) && (0..<100000).contains(b.batch) && (0...2_000_000).contains(b.presentElements)
                && (0...b.presentElements).contains(b.nonfiniteElements) && (0...(b.presentElements-b.nonfiniteElements)).contains(b.zeroElements)
                && (0...256).contains(b.missingParameters)
                && (b.nonfiniteElements>0 || b.presentElements==0 ? b.l2Norm==nil : b.l2Norm.map{$0.isFinite && $0>=0} == true)
              }),Set(activations.map(\.nodeID)).count==activations.count,
              activations.allSatisfy({!$0.nodeID.isEmpty && $0.nodeID.count<=256 && !$0.component.isEmpty && $0.component.count<=128 && $0.values.count<=8 && $0.values.allSatisfy(\.isFinite)}) else {throw DiagnosticContractError("Measured counts or samples violate their capture contract.")}
    }
}
struct MeasuredDiagnosticRule:Equatable {
    let id:String
    let family:String
    let title:String
    let prerequisites:[String]
    let thresholds:[String:Double]
    let confounders:[String]
    let experiment:String
    let lesson:String
}
struct MeasuredDiagnosticFinding:Codable,Equatable,Identifiable {
    var id:String {ruleID+(nodeID.map{":"+$0} ?? "")}
    let ruleID:String
    let nodeID:String?
    let observation:String
    let interpretation:String
    let references:[String]
    let values:[String:Double]
}
struct MeasuredDiagnosticCheck:Codable,Equatable {
    let ruleID:String
    let status:DiagnosticEvaluationStatus
    let reason:String
}
struct MeasuredDiagnosticReport:Codable,Equatable {
    var schemaVersion=1
    var registryID="kwip.measured-diagnostics"
    var registryVersion=1
    let runID:String
    let requestSHA256:String
    let inputSHA256:String
    let findings:[MeasuredDiagnosticFinding]
    let checks:[MeasuredDiagnosticCheck]
    func validate()throws {
        let ids=try MeasuredDiagnostics.rules(forVersion:registryVersion).map(\.id)
        guard schemaVersion==1,registryID=="kwip.measured-diagnostics",UUID(uuidString:runID) != nil,
              WorkflowArchiveCodec.isHash(requestSHA256),WorkflowArchiveCodec.isHash(inputSHA256),checks.map(\.ruleID)==ids,
              findings.count<=260,Set(findings.map(\.id)).count==findings.count,
              Set(checks.filter{$0.status == .emitted}.map(\.ruleID))==Set(findings.map(\.ruleID)),
              checks.allSatisfy({!$0.reason.isEmpty && $0.reason.count<=1024}),
              findings.allSatisfy({ids.contains($0.ruleID) && !$0.observation.isEmpty && !$0.interpretation.isEmpty && !$0.references.isEmpty && $0.references.count<=1024 && $0.values.values.allSatisfy(\.isFinite)}) else {throw DiagnosticContractError("Invalid measured diagnostic report or unsupported ruleset.")}
    }
    static func decode(_ data:Data)throws->Self {
        let object=try DiagnosticSchema.object(data)
        try DiagnosticSchema.keys(object,["schemaVersion","registryID","registryVersion","runID","requestSHA256","inputSHA256","findings","checks"])
        for item in object["checks"] as? [[String:Any]] ?? [] {try DiagnosticSchema.keys(item,["ruleID","status","reason"])}
        for item in object["findings"] as? [[String:Any]] ?? [] {try DiagnosticSchema.keys(item,["ruleID","nodeID","observation","interpretation","references","values"])}
        let value=try JSONDecoder().decode(Self.self,from:data);try value.validate();return value
    }
}
