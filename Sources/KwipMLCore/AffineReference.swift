import Foundation

/// Public original-map data for the exercise, never the composed solution.
public struct AffineReferenceMap: Equatable {
  public let name: String
  public let weight: LearningTensor
  public let bias: LearningTensor
  public var inputDimension: Int { weight.shape[1] }
  public var outputDimension: Int { weight.shape[0] }

  public init(name: String, weight: LearningTensor, bias: LearningTensor) throws {
    guard !name.isEmpty, name.count <= 128, weight.wellFormed, bias.wellFormed,
      weight.shape.count == 2, bias.shape == [weight.shape[0]]
    else {
      throw LearningContractError(
        "An affine reference requires finite [out,in] weights and a matching [out] bias.")
    }
    self.name = name
    self.weight = weight
    self.bias = bias
  }

  public func apply(_ input: [Double]) throws -> [Double] {
    guard input.count == inputDimension, input.allSatisfy(\.isFinite) else {
      throw LearningContractError(
        "Reference input must have the declared width and finite coordinates.")
    }
    let output = (0..<outputDimension).map { row in
      (0..<inputDimension).reduce(bias.values[row]) { value, column in
        value + weight.values[row * inputDimension + column] * input[column]
      }
    }
    guard output.allSatisfy(\.isFinite) else {
      throw LearningContractError("Reference calculation exceeded finite bounds.")
    }
    return output
  }
}

public struct AffineCompositionReference: Equatable {
  public let first: AffineReferenceMap
  public let second: AffineReferenceMap

  public init(first: AffineReferenceMap, second: AffineReferenceMap) throws {
    guard first.outputDimension == second.inputDimension else {
      throw LearningContractError(
        "The first affine map's output width must equal the second map's input width.")
    }
    self.first = first
    self.second = second
  }

  public func apply(_ input: [Double]) throws -> [Double] { try second.apply(first.apply(input)) }

  private var opening: String {
    let dimension = first.inputDimension == 2 ? "R²" : "R^\(first.inputDimension)"
    return "For x in \(dimension) reproduce W2(W1x+b1)+b2 using one or more affine Linear maps."
  }
  private var guidance: String {
    "Predict dimensions and derive the combined bias before building. Any equivalent construction may pass; no specific node names or layout are required."
  }
  /// Prose for an interface that displays the original maps as separate grids.
  public var instructions: String { opening + " " + guidance }
  /// Plain-text equivalent for consumers without matrix grids.
  public var taskText: String {
    opening
      + " W1=\(matrix(first.weight)), b1=\(vector(first.bias.values)), W2=\(matrix(second.weight)), b2=\(vector(second.bias.values)). "
      + guidance
  }
  private func vector(_ values: [Double]) -> String {
    "["
      + values.map { value in
        let text = String(value)
        return text.hasSuffix(".0") ? String(text.dropLast(2)) : text
      }.joined(separator: ",") + "]"
  }
  private func matrix(_ value: LearningTensor) -> String {
    "["
      + (0..<value.shape[0]).map { row in
        vector(Array(value.values[(row * value.shape[1])..<((row + 1) * value.shape[1])]))
      }.joined(separator: ",") + "]"
  }
}

extension LearningExamples {
  public static let affineReference: AffineCompositionReference = {
    // Fixed application-owned constants validated once; public constructors reject malformed alternatives.
    let first = try! AffineReferenceMap(
      name: "First map", weight: LearningTensor(shape: [2, 2], values: [2, -1, 0.5, 3]),
      bias: LearningTensor(shape: [2], values: [1, -2]))
    let second = try! AffineReferenceMap(
      name: "Second map", weight: LearningTensor(shape: [2, 2], values: [1, 2, -2, 1]),
      bias: LearningTensor(shape: [2], values: [0.25, 1]))
    return try! AffineCompositionReference(first: first, second: second)
  }()
}
