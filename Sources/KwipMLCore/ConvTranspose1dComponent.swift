import Foundation

/// Exact zero-padding transposed configuration is owned by the convolution bundle.
public enum ConvTranspose1dComponent {
    public static let definition = BundledConvolutionComponents.definition("conv_transpose1d")
    public static var metadata: BundledConvolutionComponents.Entry? { BundledConvolutionComponents.metadata("conv_transpose1d") }
}
