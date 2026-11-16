import Foundation

enum NumericValue: Equatable {
    case integer(Int64)
    case decimal(Double)
    var double: Double { switch self { case .integer(let value): return Double(value); case .decimal(let value): return value } }
    var text: String { switch self { case .integer(let value): return String(value); case .decimal(let value): return String(value) } }
    static func wholeDisplay(_ value: Double) -> Self { Int64(exactly: value).map(Self.integer) ?? .decimal(value) }
}

enum NumericDomain {
    case integer(minimum: Double, maximum: Double)
    case decimal(minimum: Double?, maximum: Double?)
    func parse(_ text: String) -> Result<NumericValue, NumericEditError> {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch self {
        case let .integer(minimum, maximum):
            guard let value = Int64(text), Double(value) >= minimum, Double(value) <= maximum else {
                return .failure(NumericEditError(message: "Enter a whole number from \(NumericValue.wholeDisplay(minimum).text) to \(NumericValue.wholeDisplay(maximum).text)."))
            }
            return .success(.integer(value))
        case let .decimal(minimum, maximum):
            guard let value = Double(text), value.isFinite else { return .failure(NumericEditError(message: "Enter a finite number.")) }
            guard minimum.map({ value >= $0 }) ?? true, maximum.map({ value <= $0 }) ?? true else { return .failure(NumericEditError(message: "Enter a number within the field's declared bounds.")) }
            return .success(.decimal(value))
        }
    }
}
struct NumericEditError: Error { let message: String }

/// Pure commit state shared by Return, Apply and focus-loss routes.
struct NumericEditState {
    var text: String
    var error: String?
    private var committedText: String
    var needsCommit: Bool { text != committedText || error != nil }
    init(value: NumericValue, pendingText: String? = nil) { text = pendingText ?? value.text; committedText = value.text }
    mutating func commit(domain: NumericDomain, origin: String, currentOrigin: String?, apply: (NumericValue) -> Bool) -> Bool {
        guard currentOrigin == origin else { error = "Selection changed. This value was not applied."; return false }
        switch domain.parse(text) {
        case .failure(let failure): error = "Not applied. " + failure.message; return false
        case .success(let value):
            guard apply(value) else { error = "This value was not applied. The component changed or rejected the edit."; return false }
            text = value.text; committedText = value.text; error = nil; return true
        }
    }
}

/// Removal clears the gate once; late focus callbacks must not resurrect a removed field.
struct NumericDraftLifecycle {
    struct Report { let origin: String; let pending: Bool }
    private(set) var isMounted = false
    mutating func appear() { isMounted = true }
    func report(origin: String, pending: Bool) -> Report? { isMounted ? Report(origin: origin, pending: pending) : nil }
    mutating func disappear(origin: String) -> Report { isMounted = false; return Report(origin: origin, pending: false) }
}
