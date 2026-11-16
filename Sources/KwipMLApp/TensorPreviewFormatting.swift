import Foundation
import KwipMLCore

/// Never pass integer indices through Double: values above 2^53 would change.
enum TensorPreviewFormatting {
    static func selectedOutputPort(available:[String],preferred:String?)->String? {
        if let preferred,available.contains(preferred) {return preferred}
        return available.contains("out") ? "out":available.sorted().first
    }

    static func outputs(_ preview: [String: Any]) -> [String: [String: Any]] {
        guard preview["outputs"] != nil else { return ["out": preview] }
        guard let ports = preview["outputs"] as? [String: [String: Any]], !ports.isEmpty, ports.count <= 16 else { return [:] }
        return ports.mapValues { value in
            var child = value
            if let status = preview["captureStatus"] { child["captureStatus"] = status }
            return child
        }
    }
    static func output(_ preview: [String: Any], port: String) -> [String: Any]? {
        outputs(preview)[port]
    }

    static func values(_ preview:[String:Any])->String {
        guard let dtype=preview["dtype"] as? String,["int32","int64","float32","float64"].contains(dtype) else {
            return "Unsupported captured type"
        }
        guard let raw=preview["values"] as? [Any] else {return "No captured values"}
        if let status = preview["captureStatus"] {
            guard let status = status as? String, ["captured", "omittedByPolicy"].contains(status) else { return "Invalid capture status" }
            if status == "omittedByPolicy" { return raw.isEmpty ? "Not captured by this run’s policy" : "Invalid captured values" }
        }
        let prefix = Array(raw.prefix(8))
        guard prefix.allSatisfy({ $0 is NSNumber }), JSONSerialization.isValidJSONObject(prefix),
              let data=try? JSONSerialization.data(withJSONObject:prefix),
              let values=try? JSONDecoder().decode([JSONValue].self,from:data) else {return "Invalid captured values"}
        var formatted:[String]=[]
        for value in values {
            if dtype=="int64" || dtype=="int32" {
                guard case .integer(let integer)=value,
                      dtype != "int32" || (Int64(Int32.min)...Int64(Int32.max)).contains(integer) else {return "Invalid captured values"}
                formatted.append(String(integer))
            } else {
                guard let number=value.doubleValue,number.isFinite else {return "Invalid captured values"}
                formatted.append(String(format:"%.3f",number))
            }
        }
        return formatted.isEmpty ? "No captured values" : formatted.joined(separator:"   ")
    }
}
