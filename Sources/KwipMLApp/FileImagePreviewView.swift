import SwiftUI
import AppKit
import KwipMLCore

/// Read-only exploration of verified captured pixels. Display scaling never
/// changes the table reference or its immutable pixel artifact.
struct FileImagePreviewView: View {
    let reference: FileImageReference
    let resolver: FileImageArtifactResolver
    var ocrRegions:[OCRRegionPresentation]=[]
    var embedded=false
    @State private var selectedOCR:Int?
    @State private var showRegions=true
    @State private var zoom:Double=1
    @Environment(\.dismiss) private var dismiss
    @State private var decoded: FileImageReference.Decoded?
    @State private var thumbnail: CGImage?
    @State private var error = ""
    @State private var x = 0
    @State private var y = 0
    @State private var selectedRow: Int?
    @State private var xDraft = "0"
    @State private var yDraft = "0"
    @State private var navigationError = ""
    @State private var informationExpanded = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 7) {
                    Label("Captured image", systemImage: "photo").font(.headline).foregroundStyle(MLTheme.teal)
                    HStack(spacing: 6) {
                        chip("\(reference.width) × \(reference.height)")
                        chip("\(reference.channels) channels")
                        chip(reference.dtype)
                        chip(reference.layout)
                        if let mode = decoded?.policy["actualMode"] { chip(mode) }
                    }
                }
                Spacer()
                if !embedded {Button("Close") { dismiss() }.buttonStyle(FlatButtonStyle())}
            }
            if let decoded {
                HStack(alignment: .top, spacing: 14) {
                    Group {if ocrRegions.isEmpty {displayPreview} else {ocrPreview}}.frame(minWidth:280,maxWidth:.infinity)
                    VStack(alignment: .leading, spacing: 9) {
                        if !ocrRegions.isEmpty {ocrDetail}
                        HStack {
                            Text("Pixel values").font(.subheadline.weight(.semibold))
                            Spacer()
                            Text(windowLabel).font(.system(size: 11, design: .monospaced)).foregroundStyle(MLTheme.muted)
                        }
                        HStack(alignment: .bottom, spacing: 8) {
                            coordinateField("Column x", text: $xDraft, maximum: reference.width - 1)
                            coordinateField("Row y", text: $yDraft, maximum: reference.height - 1)
                            Button("Go") { jumpToPixel() }.buttonStyle(FlatButtonStyle(prominent: true)).padding(.bottom, 1)
                        }
                        if !navigationError.isEmpty { Text(navigationError).font(.caption).foregroundStyle(MLTheme.rose) }
                        CapturedNativeTable(columns: pixelColumns, rows: pixelRows(decoded), selectedRow: $selectedRow, rowHeaderTitle: "Row")
                            .overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }.frame(height: 420)
                DisclosureGroup(isExpanded: $informationExpanded) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Coordinates are zero-based. Table row labels are one-based. Each cell lists channels in captured order; the visible window contains at most 8 × 8 pixels.")
                        Text(thumbnail == nil
                            ? "This mode is shown numerically. No display normalization or color conversion is applied."
                            : "The display preview uses a thumbnail bounded to 512 pixels per side, with nearest-pixel sampling. Its checkerboard reveals transparency. Captured pixel values are unchanged.")
                        HStack(alignment: .top, spacing: 18) {
                            ForEach(decoded.policy.keys.sorted(), id: \.self) { key in
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(policyTitle(key)).foregroundStyle(MLTheme.muted)
                                    Text(decoded.policy[key]!).textSelection(.enabled)
                                }
                            }
                        }
                    }.font(.caption).padding(.top, 8)
                } label: {
                    Label("Information", systemImage: "info.circle").font(.caption.weight(.semibold))
                }.tint(MLTheme.teal)
                Spacer(minLength: 0)
            } else if error.isEmpty {
                ProgressView("Verifying captured image…").frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Text(error).foregroundStyle(MLTheme.rose).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
        }.padding(20).frame(width: embedded ? nil:1000, height: embedded ? nil:700)
        .task(id: reference.sha256) { await load() }
        .onExitCommand {if !embedded {dismiss()}}
    }
    private var ocrPreview:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {Text("Source image & OCR regions").font(.subheadline.weight(.semibold));Spacer();Button("Fit"){zoom=1}.buttonStyle(FlatButtonStyle())}
            HStack {Toggle("Show regions",isOn:$showRegions);Slider(value:$zoom,in:0.5...4).accessibilityLabel("OCR image zoom");Text(String(format:"%.0f%%",zoom*100)).font(.caption)}
            GeometryReader {viewport in
                let scale=min(viewport.size.width/Double(reference.width),viewport.size.height/Double(reference.height))*zoom
                ScrollView([.horizontal,.vertical]) {
                    ZStack(alignment:.topLeading) {
                        if let thumbnail {Image(decorative:thumbnail,scale:1).resizable().interpolation(.none).frame(width:Double(reference.width)*scale,height:Double(reference.height)*scale)}
                        if showRegions {ForEach(ocrRegions) {region in
                            let x=region.points.map(\.x).min() ?? 0,y=region.points.map(\.y).min() ?? 0
                            let w=(region.points.map(\.x).max() ?? 0)-x,h=(region.points.map(\.y).max() ?? 0)-y
                            Path {path in for (i,p) in region.points.enumerated() {let point=CGPoint(x:p.x*scale,y:p.y*scale);if i==0 {path.move(to:point)} else {path.addLine(to:point)}};path.closeSubpath()}.stroke(selectedOCR==region.id ? MLTheme.rose:MLTheme.teal,lineWidth:2)
                            Button {selectedOCR=region.id} label:{Color.clear.frame(width:max(12,w*scale),height:max(12,h*scale))}.buttonStyle(.plain).offset(x:x*scale,y:y*scale).accessibilityLabel("OCR region \(region.regionNumber+1): \(region.text)")
                        }}
                    }.frame(width:Double(reference.width)*scale,height:Double(reference.height)*scale).frame(minWidth:viewport.size.width,minHeight:viewport.size.height)
                }
            }.overlay(Rectangle().stroke(MLTheme.rule))
            Text("Source-image pixel coordinates · overlays from the current table page only (\(ocrRegions.count) regions). Sorting/filtering changes coverage, not extracted text.").font(.caption2).foregroundStyle(MLTheme.muted)
        }
    }
    private var ocrDetail:some View {
        VStack(alignment:.leading,spacing:8) {
            if let region=ocrRegions.first(where:{$0.id==selectedOCR}) ?? ocrRegions.first {
                Text("Region \(region.regionNumber+1) · recognized text").font(.subheadline.weight(.semibold))
                Text(region.text).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(10).overlay(Rectangle().stroke(MLTheme.rule))
                Text(String(format:"Model confidence %.3f",region.confidence)).font(.caption)
                if region.review != "none" {Label("Review suggested",systemImage:"exclamationmark.triangle").font(.caption).foregroundStyle(.orange)}
                Text("Read-only captured result. Review flags do not rewrite the text.").font(.caption2).foregroundStyle(MLTheme.muted)
            }
            Divider()
        }
    }
    private func chip(_ value: String) -> some View {
        Text(value).font(.system(size: 11, weight: .medium, design: .monospaced))
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(MLTheme.teal.opacity(0.08)).overlay(RoundedRectangle(cornerRadius: 4).stroke(MLTheme.teal.opacity(0.25)))
            .clipShape(RoundedRectangle(cornerRadius: 4))
    }
    private var displayPreview: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Display preview").font(.subheadline.weight(.semibold))
                Spacer()
                Label("Verified capture", systemImage: "checkmark.shield").font(.caption2).foregroundStyle(MLTheme.teal)
            }
            ZStack {
                FileImageCheckerboard()
                if let thumbnail {
                    Image(decorative: thumbnail, scale: 1).resizable().interpolation(.none).scaledToFit().padding(16)
                } else {
                    VStack(spacing: 9) {
                        Image(systemName: "number.square").font(.system(size: 30)).foregroundStyle(MLTheme.teal)
                        Text("Raw pixel mode").font(.subheadline.weight(.semibold))
                        Text("Explore the captured values alongside.").font(.caption).foregroundStyle(MLTheme.muted)
                    }.padding(16).background(MLTheme.panel.opacity(0.95))
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(MLTheme.rule, lineWidth: 1))
        }
    }
    private func coordinateField(_ title: String, text: Binding<String>, maximum: Int) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack { Text(title).font(.caption.weight(.medium)); Text("0–\(maximum)").font(.caption2).foregroundStyle(MLTheme.muted) }
            TextField("0", text: text).flatField().accessibilityLabel(title).onSubmit(jumpToPixel)
        }.frame(maxWidth: .infinity)
    }
    private var windowLabel: String {
        "x \(x)–\(min(x + 7, reference.width - 1)) · y \(y)–\(min(y + 7, reference.height - 1))"
    }
    private func jumpToPixel() {
        guard let column = FileImagePixelNavigation.coordinate(xDraft, limit: reference.width),
              let row = FileImagePixelNavigation.coordinate(yDraft, limit: reference.height) else {
            navigationError = "Enter whole-number coordinates within the ranges above."; return
        }
        x = column; y = row; xDraft = String(column); yDraft = String(row)
        selectedRow = nil; navigationError = ""
    }
    private func policyTitle(_ key: String) -> String {
        ["orientation": "Orientation", "colorMode": "Color policy", "palette": "Palette", "actualMode": "Captured mode"][key] ?? key
    }
    private var pixelColumns: [CapturedDataGrid.Column] {
        (x..<min(x + 8, reference.width)).enumerated().map { index, position in
            .init(id: index, name: String(position), label: "x = \(position)", dtype: reference.dtype, nullability: "required")
        }
    }
    private func pixelRows(_ value: FileImageReference.Decoded) -> [CapturedDataGrid.Row] {
        (y..<min(y + 8, reference.height)).map { row in
            .init(index: row, cells: (x..<min(x + 8, reference.width)).map { column in
                .init(text: FileImagePixels.values(value, x: column, y: row).joined(separator: ", "), isAbsent: false)
            })
        }
    }
    @MainActor private func load() async {
        decoded = nil; thumbnail = nil; error = ""; x = 0; y = 0; selectedRow = nil
        xDraft = "0"; yDraft = "0"; navigationError = ""; informationExpanded = false
        let job = Task.detached(priority: .userInitiated) { () throws -> (FileImageReference.Decoded, CGImage?) in
            let pixels = try resolver.read(reference)
            return (pixels, try FileImagePixels.thumbnail(pixels))
        }
        do {
            let result = try await withTaskCancellationHandler(operation: { try await job.value }, onCancel: { job.cancel() })
            guard !Task.isCancelled else { return }
            decoded = result.0; thumbnail = result.1
        } catch {
            guard !Task.isCancelled else { return }
            self.error = "Image could not be verified: " + error.localizedDescription
        }
    }
}

enum FileImagePixelNavigation {
    static func coordinate(_ text: String, limit: Int) -> Int? {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard limit > 0, !value.isEmpty, value.utf8.count <= 10, value.utf8.allSatisfy({ (48...57).contains($0) }),
              let number = Int(value), (0..<limit).contains(number) else { return nil }
        return number
    }
}

private struct FileImageCheckerboard: View {
    var body: some View {
        Canvas { context, size in
            let step: CGFloat = 16
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(nsColor: .controlBackgroundColor)))
            for row in 0..<Int(ceil(size.height / step)) {
                for column in 0..<Int(ceil(size.width / step)) where (row + column).isMultiple(of: 2) {
                    context.fill(Path(CGRect(x: CGFloat(column) * step, y: CGFloat(row) * step, width: step, height: step)), with: .color(MLTheme.muted.opacity(0.10)))
                }
            }
        }.accessibilityHidden(true)
    }
}

enum FileImagePixels {
    static func values(_ value: FileImageReference.Decoded, x: Int, y: Int) -> [String] {
        let ref = value.reference
        guard (0..<ref.width).contains(x), (0..<ref.height).contains(y) else { return [] }
        let stride = ref.dtype == "uint8" ? 1 : (ref.dtype == "uint16" ? 2 : 4)
        return (0..<ref.channels).map { channel in
            let ordinal = ref.layout == "HWC" ? (y * ref.width + x) * ref.channels + channel : channel * ref.width * ref.height + y * ref.width + x
            let offset = ordinal * stride
            var bits: UInt32 = 0
            for byte in 0..<stride { bits |= UInt32(value.pixels[offset + byte]) << (8 * byte) }
            switch ref.dtype {
            case "uint8", "uint16": return String(bits)
            case "int32": return String(Int32(bitPattern: bits))
            case "float32": return String(Float(bitPattern: bits))
            default: return "unsupported"
            }
        }
    }
    static func thumbnail(_ value: FileImageReference.Decoded) throws -> CGImage? {
        let ref = value.reference
        guard ref.dtype == "uint8", ["L", "LA", "RGB", "RGBA"].contains(value.policy["actualMode"] ?? "") else { return nil }
        let scale = min(1.0, 512.0 / Double(max(ref.width, ref.height)))
        let width = max(1, Int(Double(ref.width) * scale)), height = max(1, Int(Double(ref.height) * scale))
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            try Task.checkCancellation()
            for x in 0..<width {
                let sourceX = min(ref.width - 1, x * ref.width / width), sourceY = min(ref.height - 1, y * ref.height / height)
                func channel(_ c: Int) -> UInt8 {
                    let position = ref.layout == "HWC" ? (sourceY * ref.width + sourceX) * ref.channels + c : c * ref.width * ref.height + sourceY * ref.width + sourceX
                    return value.pixels[position]
                }
                let offset = (y * width + x) * 4
                pixels[offset] = channel(0)
                pixels[offset + 1] = ref.channels <= 2 ? channel(0) : channel(1)
                pixels[offset + 2] = ref.channels <= 2 ? channel(0) : channel(2)
                pixels[offset + 3] = ref.channels == 2 ? channel(1) : (ref.channels == 4 ? channel(3) : 255)
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
