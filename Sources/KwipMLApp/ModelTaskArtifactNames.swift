import Foundation

enum ModelTaskArtifactNames {
    static func continuationTensor(_ name:String)->Bool {
        name.range(of:#"^training-state(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors$"#,options:.regularExpression) != nil
    }
}
