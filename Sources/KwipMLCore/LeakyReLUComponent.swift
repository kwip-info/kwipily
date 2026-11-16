import Foundation

/// Reviewed out-of-place activation definition from the authored bundle.
public enum LeakyReLUComponent {
    public static let definition = BundledActivationComponents.definition("leaky_relu")
    public static var metadata: BundledActivationComponents.Entry? { BundledActivationComponents.entries["leaky_relu"] }
}
