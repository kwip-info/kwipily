import Foundation
import KwipMLCore

/// Complete population, authenticated by the completed training/evaluation receipt.
struct WorkflowPredictionTableEvidence:Equatable {
    let file:WorkflowArchiveFile
    let partition:String
    let rowCount:Int
    let columns:[[String:JSONValue]]
    var outputMetadata:JSONValue {.object(["kind":.string("table"),"preview":.object(["columns":.array(columns.map(JSONValue.object)),"rows":.array([]),"rowCount":.integer(Int64(rowCount)),"partition":.string(partition)])])}
    static func verify(_ bytes:Data,file:WorkflowArchiveFile,partition:String,observationIDs:[String],sourceDigest:String?=nil)throws->Self {
        func refuse()->WorkflowArchiveError {.init("Prediction table differs from its verified population, schema or artifact pin.")}
        guard ["validation","test"].contains(partition),file.bytes==bytes.count,bytes.count<=50_000_000,
              WorkflowArchiveCodec.hash(bytes)==file.sha256,!observationIDs.isEmpty,Set(observationIDs).count==observationIDs.count else {throw refuse()}
        let table=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard Set(table.keys)==["schemaVersion","kind","columns","rows","lineage"],table["schemaVersion"]==1,table["kind"] == .string("table"),
              case .array(let columns)=table["columns"],(4...128).contains(columns.count),
              case .array(let rows)=table["rows"],rows.count==observationIDs.count,rows.count<=5_000_000/columns.count,
              case .object(let lineage)=table["lineage"],lineage["role"] == .string(partition),lineage["observationIDs"] == .array(observationIDs.map(JSONValue.string)) else {throw refuse()}
        if let sourceDigest,lineage["sourceDigest"] != .string(sourceDigest) {throw refuse()}
        var names=Set<String>()
        for column in columns {
            guard case .object(let value)=column,Set(value.keys)==["name","dtype","nullable"],let name=value["name"]?.stringValue,names.insert(name).inserted,
                  ["string","int64","float64"].contains(value["dtype"]?.stringValue ?? ""),value["nullable"] == false else {throw refuse()}
        }
        guard names.isSuperset(of:["observation_id","partition","target","output"]) else {throw refuse()}
        for (index,row) in rows.enumerated() {
            if index%256==0 {try Task.checkCancellation()}
            guard case .array(let cells)=row,cells.count==columns.count,cells[0] == .string(observationIDs[index]),cells[1] == .string(partition) else {throw refuse()}
        }
        return .init(file:file,partition:partition,rowCount:rows.count,columns:columns.compactMap {if case .object(let value)=$0 {return value};return nil})
    }
    static func verifyClassification(_ bytes:Data,rows:[WorkflowEvaluationPrediction])throws {
        let table=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard case .array(let columns)=table["columns"],case .array(let values)=table["rows"],values.count==rows.count,
              columns.compactMap({if case .object(let value)=$0 {return value["name"]?.stringValue};return nil}) == ["observation_id","partition","target","predicted","probability_0","probability_1","output"] else {throw WorkflowArchiveError("Prediction table classification schema differs from verified evaluation.")}
        for (i,value) in values.enumerated() {
            if i%256==0 {try Task.checkCancellation()}
            let expected=rows[i]
            guard case .array(let cells)=value,cells.count==7,cells[0] == .string(expected.observationID),cells[2].integerValue==expected.target,cells[3].integerValue==expected.predicted,
                  let output=cells[6].stringValue,let actual=try? JSONDecoder().decode([Double].self,from:Data(output.utf8)),actual==expected.logits else {throw WorkflowArchiveError("Prediction table values differ from the full verified evaluation.")}
            let peak=expected.logits.max()!,weights=expected.logits.map {exp($0-peak)},total=weights.reduce(0,+)
            for j in 0..<2 {guard let probability=cells[4+j].doubleValue,probability.isFinite,abs(probability-weights[j]/total)<=1e-12 else {throw WorkflowArchiveError("Prediction table probabilities differ from its verified class scores.")}}
        }
    }
    func artifact(directory:URL)->TableViewQuerySession.Artifact {
        .init(indexURL:directory.appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table")
    }
}
