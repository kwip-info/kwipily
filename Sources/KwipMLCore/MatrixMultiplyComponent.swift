import Foundation

/// The authored Tensor math bundle owns the parameterless two-input definition.
public enum MatrixMultiplyComponent {
    public static let definition = BundledTensorMathComponents.definition("matmul")
    public static var metadata: BundledTensorMathComponents.Entry? { BundledTensorMathComponents.entries["matmul"] }
}
