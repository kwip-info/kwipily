import Foundation

/// Exact zero-padding transposed configuration is owned by the convolution bundle.
public enum ConvTranspose3dComponent {
    public static let definition = BundledConvolutionComponents.definition("conv_transpose3d")
    public static var metadata: BundledConvolutionComponents.Entry? { BundledConvolutionComponents.metadata("conv_transpose3d") }
}
