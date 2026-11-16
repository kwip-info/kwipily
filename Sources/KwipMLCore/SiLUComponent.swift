import Foundation

/// Reviewed out-of-place activation definition from the authored bundle.
public enum SiLUComponent {
    public static let definition = BundledActivationComponents.definition("silu")
    public static var metadata: BundledActivationComponents.Entry? { BundledActivationComponents.entries["silu"] }
}
