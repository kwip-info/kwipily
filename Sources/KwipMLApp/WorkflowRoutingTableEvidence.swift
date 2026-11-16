import Foundation
import KwipMLCore

/// A table is accepted only with its completed model receipt, not from a live file path.
enum WorkflowRoutingTableEvidence {
    static func verify(_ bytes:Data,file:WorkflowArchiveFile,port:String,runID:String,graphHash:String)throws->WorkflowPredictionTableEvidence {
        func fail()->WorkflowPhaseError {.init("Routing table differs from its captured model, population or bounded schema.")}
        guard ["routing","routingSample"].contains(port),bytes.count==file.bytes,bytes.count<=2_000_000,WorkflowArchiveCodec.hash(bytes)==file.sha256,
              let table=try? JSONDecoder().decode([String:JSONValue].self,from:bytes),Set(table.keys)==["schemaVersion","kind","columns","rows","lineage"],table["schemaVersion"]==1,table["kind"] == .string("table"),
              case .array(let columns)=table["columns"],case .array(let rows)=table["rows"],rows.count<=131072,
              case .object(let lineage)=table["lineage"],Set(lineage.keys)==["runID","graphHash","scope","role","sourceDigest","membershipDigest","observationIDs","dataVersion","partitionDigest","fittedStateDigests","diagnostics"],lineage["runID"] == .string(runID),lineage["graphHash"] == .string(graphHash),lineage["scope"] == .string("capturedUnitVisits") else {throw fail()}
        guard lineage["role"] == .string("unassigned"),case .array(let ids)=lineage["observationIDs"],ids.count==rows.count,Set(ids.compactMap(\.stringValue)).count==rows.count else{throw fail()}
        let names=port=="routing" ? ["block","phase","expert_id","expert","assignments","assignment_share","mean_gate_weight","routed_unit_visits"]:["block","phase","visit","unit_position","expert_ids","gate_weights"]
        let types=port=="routing" ? ["string","string","string","string","int64","float64","float64","int64"]:["string","string","int64","int64","string","string"]
        guard columns.count==names.count else {throw fail()}
        for (i,column) in columns.enumerated(){guard column == .object(["name":.string(names[i]),"dtype":.string(types[i]),"nullable":true]) else {throw fail()}}
        var groups:[String:[(Int,Double?)]]=[:],identities:Set<String>=[]
        for row in rows {
            guard case .array(let cells)=row,cells.count==names.count,let block=cells[0].stringValue,["training","inference"].contains(cells[1].stringValue ?? "") else {throw fail()}
            for (index,value) in cells.enumerated() where value != .null {
                guard types[index]=="string" ? (value.stringValue?.utf8.count ?? 4097)<=4096:types[index]=="int64" ? value.integerValue.map({$0>=0})==true:value.doubleValue.map({$0.isFinite && $0>=0 && $0<=1.0000001})==true else {throw fail()}
            }
            if port=="routing" {
                guard rows.count<=1024,let expert=cells[2].stringValue,let assignments=cells[4].integerValue,let visits=cells[7].integerValue,assignments<=visits,
                      identities.insert(block+"|"+(cells[1].stringValue ?? "")+"|"+expert).inserted else {throw fail()}
                groups[block+"|"+(cells[1].stringValue ?? ""),default:[]].append((assignments,cells[5].doubleValue))
                guard assignments==0 ? cells[6] == .null:cells[6].doubleValue != nil else {throw fail()}
            }
        }
        for values in groups.values {let total=values.reduce(0){$0+$1.0};guard (2...4).contains(values.count) else {throw fail()};for (count,share) in values {guard total==0 ? share==nil:share.map({abs($0-Double(count)/Double(total))<1e-10})==true else {throw fail()}}}
        return .init(file:file,partition:"capturedUnitVisits",rowCount:rows.count,columns:columns.compactMap{if case .object(let value)=$0{return value};return nil})
    }
}
