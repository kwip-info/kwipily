import Foundation

/// Spatial settings follow explicit axis names; indices address flattened input spatial positions.
struct SpatialMaxPoolConfiguration: Equatable, Sendable {
    let kernelSize: [Int]
    let stride: [Int]?
    let padding: [Int]
    let dilation: [Int]
    let ceilMode: Bool
    let returnIndices: Bool
    private let axes: [MaxPool1dConfiguration]
    var effectiveStride: [Int] { stride ?? kernelSize }

    private let axisNames:[String]
    private var title:String { "MaxPool\(axisNames.count)d" }

    init(values: [String:JSONValue], axisNames:[String]) throws {
        self.axisNames=axisNames
        let title="MaxPool\(axisNames.count)d"
        guard Set(values.keys) == ["kernel_size","stride","padding","dilation","ceil_mode","return_indices"] else {
            throw GraphConfigurationError("\(title) requires kernel, stride, padding, dilation, ceiling mode and return-indices settings.")
        }
        func pair(_ key:String) throws -> [JSONValue] {
            guard case .array(let items) = values[key], items.count == axisNames.count else {
                throw GraphConfigurationError("\(title) \(key) requires exactly [\(axisNames.joined(separator:", "))].")
            }
            return items
        }
        let kernel=try pair("kernel_size"), pads=try pair("padding"), dilations=try pair("dilation")
        let steps:[JSONValue]? = values["stride"] == .null ? nil : try pair("stride")
        var resolvedAxes:[MaxPool1dConfiguration]=[]
        for axis in axisNames.indices {
            do {
                resolvedAxes.append(try MaxPool1dConfiguration(values:[
                    "kernel_size":kernel[axis],"stride":steps?[axis] ?? .null,
                    "padding":pads[axis],"dilation":dilations[axis],
                    "ceil_mode":values["ceil_mode"]!,"return_indices":values["return_indices"]!]))
            } catch {
                throw GraphConfigurationError("\(title) \(axisNames[axis]): \(error.localizedDescription)")
            }
        }
        axes=resolvedAxes
        kernelSize=axes.map(\.kernelSize);padding=axes.map(\.padding);dilation=axes.map(\.dilation)
        stride=steps == nil ? nil : axes.map(\.effectiveStride)
        ceilMode=axes[0].ceilMode;returnIndices=axes[0].returnIndices
    }
    var values:[String:JSONValue] {
        func pair(_ values:[Int])->JSONValue { .array(values.map{.integer(Int64($0))}) }
        return ["kernel_size":pair(kernelSize),"stride":stride.map(pair) ?? .null,
                "padding":pair(padding),"dilation":pair(dilation),
                "ceil_mode":.bool(ceilMode),"return_indices":.bool(returnIndices)]
    }
    struct Resolved: Equatable, Sendable {
        let outputLayout:BatchedTensorLayout
        let indicesLayout:BatchedTensorLayout?
    }
    func resolved(input:BatchedTensorLayout) throws -> Resolved {
        let shape=input.sample.dimensions
        guard shape.count == axisNames.count+1 else {throw GraphConfigurationError("\(title) requires sample shape [channels, \(axisNames.joined(separator:", "))].")}
        var output=[shape[0]]
        for axis in axisNames.indices {
            do {
                let result=try axes[axis].resolved(input:.init(sample:[shape[0],shape[axis+1]]))
                output.append(result.outputLayout.sample.dimensions[1])
            } catch {
                throw GraphConfigurationError("\(title) \(axisNames[axis]): \(error.localizedDescription)")
            }
        }
        let layout=try BatchedTensorLayout(sample:output)
        return Resolved(outputLayout:layout,indicesLayout:returnIndices ? layout : nil)
    }
}
