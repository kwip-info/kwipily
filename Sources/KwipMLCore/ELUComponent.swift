import Foundation

/// Reviewed out-of-place activation definition from the authored bundle.
public enum ELUComponent {
    public static let definition = BundledActivationComponents.definition("elu")
    public static var metadata: BundledActivationComponents.Entry? { BundledActivationComponents.entries["elu"] }
}
