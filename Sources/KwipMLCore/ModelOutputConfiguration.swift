import Foundation

/// Assertions describe each observation; the leading batch dimension remains variable.
public struct ModelOutputConfiguration: Equatable {
    public var task = "single"
    public var checkExpectedOutput: Bool
    public var expectedDType: String
    public var expectedSampleRank: String
    public var expectedSampleDimensions: [Int?]
    public static let supportedDTypes = ["any", "float32", "float64", "int32", "int64", "bool"]
    public static let defaults: [String: JSONValue] = ["checkExpectedOutput": .bool(false), "expectedDType": .string("any"), "expectedSampleRank": .string("any"), "expectedSampleDimensions": .array([])]

    public init(_ config: [String: JSONValue]) throws {
        if config["task"] != nil {
            guard config == ["task":.string("gamePolicyValue"),"taskVersion":1] else {throw GraphConfigurationError("Unsupported versioned Model Output task.")}
            task="gamePolicyValue";checkExpectedOutput=false;expectedDType="float32";expectedSampleRank="exact";expectedSampleDimensions=[];return
        }
        guard Set(config.keys).isSubset(of: Set(Self.defaults.keys)) else { throw GraphConfigurationError("Model Output: unsupported expected-output setting.") }
        let value = Self.defaults.merging(config) { _, new in new }
        guard case .bool(let enabled) = value["checkExpectedOutput"],
              case .string(let dtype) = value["expectedDType"], Self.supportedDTypes.contains(dtype),
              case .string(let rank) = value["expectedSampleRank"], ["any", "exact"].contains(rank),
              case .array(let dimensions) = value["expectedSampleDimensions"], dimensions.count <= 7 else {
            throw GraphConfigurationError("Model Output: choose supported type and Any or Exact sample rank.")
        }
        checkExpectedOutput = enabled; expectedDType = dtype; expectedSampleRank = rank
        expectedSampleDimensions = try dimensions.map { dimension in
            if case .null = dimension { return nil }
            guard case .integer(let size) = dimension, (1...100_000_000).contains(size) else {
                throw GraphConfigurationError("Model Output: exact dimensions must be positive whole numbers, up to 100000000.")
            }
            return Int(size)
        }
    }
    public var values: [String: JSONValue] {
        if task == "gamePolicyValue" {return ["task":.string(task),"taskVersion":1]}
        return ["checkExpectedOutput": .bool(checkExpectedOutput), "expectedDType": .string(expectedDType), "expectedSampleRank": .string(expectedSampleRank), "expectedSampleDimensions": .array(expectedSampleDimensions.map { $0.map { .integer(Int64($0)) } ?? .null })]
    }
    public func check(sample: [Int], dtype: String) throws {
        guard checkExpectedOutput else { return }
        guard expectedDType == "any" || expectedDType == dtype else { throw GraphConfigurationError("Model Output: expected dtype \(expectedDType); observed \(dtype).") }
        guard expectedSampleRank == "exact" else { return }
        guard sample.count == expectedSampleDimensions.count else { throw GraphConfigurationError("Model Output: expected sample rank \(expectedSampleDimensions.count); observed \(sample.count).") }
        for (axis, pair) in zip(sample, expectedSampleDimensions).enumerated() {
            if let expected = pair.1, expected != pair.0 { throw GraphConfigurationError("Model Output: sample dimension \(axis + 1) expected \(expected); observed \(pair.0).") }
        }
    }
}
