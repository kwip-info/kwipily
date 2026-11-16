import SwiftUI
import KwipMLCore

/// Distinct geometry remains readable without relying on component color.
struct CanvasPortGlyphView: View {
    let glyph: CanvasPortGlyph
    var tint: Color = MLTheme.teal
    var filled = false
    var body: some View {
        Group {
            switch glyph {
            case .circle: Circle().fill(filled ? tint : MLTheme.background).overlay(Circle().stroke(tint,lineWidth:1.5))
            case .square: Rectangle().fill(filled ? tint : MLTheme.background).overlay(Rectangle().stroke(tint,lineWidth:1.5))
            case .diamond: Rectangle().fill(filled ? tint : MLTheme.background).overlay(Rectangle().stroke(tint,lineWidth:1.5)).rotationEffect(.degrees(45)).padding(2)
            case .stack: Image(systemName:"square.stack").font(.system(size:12,weight:.semibold)).foregroundStyle(tint)
            case .document: Image(systemName:"doc").font(.system(size:12,weight:.semibold)).foregroundStyle(tint)
            case .mapping: Image(systemName:"arrow.triangle.merge").font(.system(size:12,weight:.semibold)).foregroundStyle(tint)
            case .sliders: Image(systemName:"slider.horizontal.3").font(.system(size:12,weight:.semibold)).foregroundStyle(tint)
            case .lock: Image(systemName:"lock").font(.system(size:12,weight:.semibold)).foregroundStyle(tint)
            case .question: Image(systemName:"questionmark.square").font(.system(size:12,weight:.semibold)).foregroundStyle(tint)
            }
        }.frame(width:12,height:12).accessibilityHidden(true)
    }
}
