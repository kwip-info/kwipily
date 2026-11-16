import Foundation

/// Archived numerical geometry, including a true scalar []; never a batch-layout guess.
public struct ObjectiveResultTensor: Equatable {
    public let shape: [Int]
    public let dtype: ContractDType
    public let values: [Double]
    public init(json: JSONValue, shape: [Int], dtype: ContractDType) throws {
        guard shape.count <= 8, [.float32,.int64].contains(dtype) else {throw GraphConfigurationError("Unsupported objective tensor rank or dtype.")}
        var count=1
        for dimension in shape {
            guard (1...4096).contains(dimension),count<=100_000_000/dimension else {throw GraphConfigurationError("Objective tensor shape exceeds its bounded element budget.")}
            count *= dimension
        }
        var parsed:[Double]=[] // Never allocate from the untrusted declared product.
        func visit(_ value:JSONValue,_ depth:Int)throws {
            if depth < shape.count {
                guard case .array(let array)=value,array.count == shape[depth] else {throw GraphConfigurationError("Objective tensor values do not fill their exact shape.")}
                for item in array {try visit(item,depth+1)}
            } else {
                guard parsed.count<count,let number=value.doubleValue,number.isFinite else {throw GraphConfigurationError("Objective tensor contains a nonnumeric or nonfinite value.")}
                if dtype == .int64 {guard value.integerValue != nil else {throw GraphConfigurationError("Integer objective values must be exact integers.")}}
                else {guard Float(number).isFinite else {throw GraphConfigurationError("Objective value exceeds float32.")}}
                parsed.append(number)
            }
        }
        try visit(json,0)
        guard parsed.count == count else {throw GraphConfigurationError("Objective tensor is incomplete.")}
        self.shape=shape;self.dtype=dtype;values=parsed
    }
}
