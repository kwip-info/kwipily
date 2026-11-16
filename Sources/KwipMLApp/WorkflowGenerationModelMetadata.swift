import Foundation
import KwipMLCore

struct WorkflowGenerationModelMetadata:Equatable {
    let context:Int
    let vocabularySize:Int
    let hasUnknown:Bool
    let hasEOS:Bool
    static func read(_ package:WorkflowTrainedExportPackage)throws->Self {
        func object(_ name:String)throws->[String:JSONValue] {
            guard let file=package.files.first(where:{$0.path==name}) else {throw WorkflowArchiveError("Generation needs a captured next-token model and tokenizer.")}
            let bytes=try DiagnosticFileIO.read(root:package.directory,parts:[name],limit:2_000_000)
            guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==file.sha256,case .object(let values)=try JSONDecoder().decode(JSONValue.self,from:bytes) else {throw WorkflowFileLeaseError.unsafePath}
            return values
        }
        let contract=try object("input-contract.json"),tokenizers=try object("tokenizers.json")
        guard contract["schemaVersion"] == .integer(4),case .object(let input)=contract["input"],case .object(let plan)=input["plan"],plan["schemaVersion"] == .integer(2),case .array(let inputs)=plan["inputs"],inputs.count==1,case .object(let entry)=inputs[0],entry["dtype"] == .string("int64"),entry["representation"] == .string("tokens"),let name=entry["name"]?.stringValue,case .array(let shape)=entry["sampleShape"],shape.count==1,let context=shape[0].integerValue,(1...4096).contains(context),case .object(let tokenizer)=tokenizers[name],case .array(let vocabulary)=tokenizer["vocabulary"],(2...512).contains(vocabulary.count),entry["vocabularySize"] == .integer(Int64(vocabulary.count)),case .object(let configuration)=tokenizer["configuration"] else {throw WorkflowArchiveError("Generate text needs one captured int64 context and a next-token vocabulary of 2–512 entries.")}
        return .init(context:context,vocabularySize:vocabulary.count,hasUnknown:configuration["unknownPolicy"] == .string("unk"),hasEOS:configuration["eos"] == .bool(true))
    }
}
