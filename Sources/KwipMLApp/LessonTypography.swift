import SwiftUI

/// Reading preference only: never part of lesson revisions or saved graph state.
struct LessonTypography: Equatable {
    static let preferenceKey = "lessonReadingTextSize"
    static let defaultSize = 16
    static let minimumSize = 12
    static let maximumSize = 24
    let bodySize: Int
    init(bodySize: Int) { self.bodySize = Self.normalized(bodySize) }
    static func normalized(_ size: Int) -> Int { min(maximumSize, max(minimumSize, size)) }
    static func load(_ defaults: UserDefaults) -> Self {
        .init(bodySize: defaults.object(forKey: preferenceKey) == nil ? defaultSize : defaults.integer(forKey: preferenceKey))
    }
    var scale: CGFloat { CGFloat(bodySize) / 12 }
    func size(_ baseline: CGFloat) -> CGFloat { baseline * scale }
}
private struct LessonTypographyKey: EnvironmentKey {
    // References outside the reading rail keep their existing presentation.
    static let defaultValue = LessonTypography(bodySize: 12)
}
extension EnvironmentValues {
    var lessonTypography: LessonTypography {
        get { self[LessonTypographyKey.self] }
        set { self[LessonTypographyKey.self] = newValue }
    }
}
