import Foundation

/// Bounded, typed held-out observations. This is a display projection, not a dataset.
public struct PredictionPreview: Equatable {
    public struct Row: Equatable {
        public let inputs: [String]
        public let label: Int
        public let predicted: Int
    }
    public enum CaptureStatus: String, Decodable { case captured, omittedByPolicy }
    public let captureStatus: CaptureStatus
    public let dtype: String
    public let sampleShape: [Int]
    public let coordinates: [String]
    public let rows: [Row]
    public let evaluationCount: Int

    public static func decode(_ data: Data) throws -> Self {
        struct Envelope: Decodable {
            struct Input: Decodable { let dtype: String; let sampleShape: [Int] }
            struct Observation: Decodable { let input: JSONValue; let label: Int; let predicted: Int }
            let sampleCapture: TrainingSampleCapturePolicy?
            let schemaVersion: Int?
            let captureStatus: CaptureStatus?
            let predictionInput: Input
            let predictionRows: [Observation]
            let evaluationCount: Int
        }
        func invalid() -> GraphConfigurationError { .init("Prediction preview has invalid dimensions, values or labels.") }
        guard data.count <= 1_000_000 else { throw invalid() }
        let fields = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        if fields?["sampleCapture"] is NSNull || fields?["captureStatus"] is NSNull { throw invalid() }
        let envelope = try JSONDecoder().decode(Envelope.self, from: data)
        let status: CaptureStatus
        if let policy = envelope.sampleCapture {
            guard envelope.schemaVersion == nil, envelope.captureStatus == nil else { throw invalid() }
            status = policy.predictionRows ? .captured : .omittedByPolicy
        } else if envelope.schemaVersion == 2 {
            guard let captured = envelope.captureStatus else { throw invalid() }
            status = captured
        } else {
            guard envelope.schemaVersion == nil || envelope.schemaVersion == 1, envelope.captureStatus == nil else { throw invalid() }
            status = .captured
        }
        let validRows = status == .omittedByPolicy ? envelope.predictionRows.isEmpty : (1...min(200, max(1, envelope.evaluationCount))).contains(envelope.predictionRows.count)
        let shape = envelope.predictionInput.sampleShape, dtype = envelope.predictionInput.dtype
        guard ["int64", "float32"].contains(dtype), (1...7).contains(shape.count),
              (1...100_000).contains(envelope.evaluationCount),
              validRows else { throw invalid() }
        var count = 1
        for dimension in shape {
            guard dimension > 0, dimension <= 128 / count else { throw invalid() }
            count *= dimension
        }
        func values(_ value: JSONValue, axis: Int) throws -> [String] {
            if axis < shape.count {
                guard case .array(let children) = value, children.count == shape[axis] else { throw invalid() }
                return try children.flatMap { try values($0, axis: axis + 1) }
            }
            if dtype == "int64" {
                guard case .integer(let integer) = value else { throw invalid() }
                return [String(integer)]
            }
            guard let number = value.doubleValue, number.isFinite, abs(number) <= Double(Float.greatestFiniteMagnitude) else { throw invalid() }
            return [String(format: "%.6g", number)]
        }
        let rows = try envelope.predictionRows.map { row -> Row in
            guard (0...1).contains(row.label), (0...1).contains(row.predicted) else { throw invalid() }
            return try Row(inputs: values(row.input, axis: 0), label: row.label, predicted: row.predicted)
        }
        let coordinates = (0..<count).map { flat -> String in
            var remainder = flat, indices = Array(repeating: 0, count: shape.count)
            for axis in shape.indices.reversed() { indices[axis] = remainder % shape[axis]; remainder /= shape[axis] }
            return "[" + indices.map(String.init).joined(separator: ",") + "]"
        }
        return Self(captureStatus: status, dtype: dtype, sampleShape: shape, coordinates: coordinates, rows: rows, evaluationCount: envelope.evaluationCount)
    }
}

public extension PredictionPreview {
    /// Version4 binds every named source independently; presentation flattens only
    /// these bounded captured values, never the model's actual input contract.
    static func decodeNamed(_ data:Data,graph:GraphDocument)throws->Self {
        func refuse()->GraphConfigurationError {.init("Named prediction values differ from the captured model inputs.")}
        guard data.count<=1000000,let names=graph.namedInputs,!names.isEmpty else {throw refuse()}
        guard case .object(let root)=try JSONDecoder().decode(JSONValue.self,from:data),root["schemaVersion"] == 4,
              let statusText=root["captureStatus"]?.stringValue,let status=CaptureStatus(rawValue:statusText),
              let count=root["evaluationCount"]?.integerValue,(1...100000).contains(count),
              case .object(let contract)=root["predictionInput"],contract["kind"] == .string("namedInputs"),contract["schemaVersion"] == 1,
              case .object(let declarations)=contract["inputs"],Set(declarations.keys)==Set(names.keys),
              case .array(let values)=root["predictionRows"],values.count==(status == .captured ? min(200,count):0) else {throw refuse()}
        var shapes:[String:[Int]]=[:],types:[String:ContractDType]=[:],coordinates=[String]()
        for name in names.keys.sorted() {
            guard let node=graph.nodes.first(where:{$0.id==names[name]}),["input","index_input"].contains(node.type) else {throw refuse()}
            let tensor:TypedTensorContract
            if node.type=="index_input" {tensor=try IndexInputConfiguration(values:node.config).tensor}
            else {guard let width=node.config["features"]?.integerValue,(1...512).contains(width) else {throw refuse()};tensor=try .init(sample:[width],dtype:.float32)}
            let shape=tensor.layout.sample.dimensions
            guard tensor.layout.sample.elementCount<=1024,case .object(let declared)=declarations[name],declared["nodeID"] == .string(node.id),declared["dtype"] == .string(tensor.dtype.rawValue),declared["sampleShape"] == .array(shape.map {.integer(Int64($0))}) else {throw refuse()}
            shapes[name]=shape;types[name]=tensor.dtype
            coordinates += (0..<tensor.layout.sample.elementCount).map {name+"["+String($0)+"]"}
        }
        func flatten(_ value:JSONValue,shape:[Int],dtype:ContractDType)throws->[String] {
            guard let first=shape.first else {
                if dtype == .int64 {guard case .integer(let i)=value else {throw refuse()};return [String(i)]}
                guard let d=value.doubleValue,d.isFinite,abs(d)<=Double(Float.greatestFiniteMagnitude) else {throw refuse()};return [String(format:"%.6g",d)]
            }
            guard case .array(let values)=value,values.count==first else {throw refuse()}
            return try values.flatMap {try flatten($0,shape:Array(shape.dropFirst()),dtype:dtype)}
        }
        let rows=try values.map {value->Row in
            guard case .object(let row)=value,Set(row.keys)==["input","label","predicted"],case .object(let inputs)=row["input"],Set(inputs.keys)==Set(names.keys),let label=row["label"]?.integerValue,let predicted=row["predicted"]?.integerValue,(0...1).contains(label),(0...1).contains(predicted) else {throw refuse()}
            let fields=try names.keys.sorted().flatMap {try flatten(inputs[$0]!,shape:shapes[$0]!,dtype:types[$0]!)}
            return .init(inputs:fields,label:label,predicted:predicted)
        }
        return .init(captureStatus:status,dtype:"named",sampleShape:[coordinates.count],coordinates:coordinates,rows:rows,evaluationCount:count)
    }
}
