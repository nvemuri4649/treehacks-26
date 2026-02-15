// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Cena",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "Cena",
            targets: ["Cena"]
        )
    ],
    targets: [
        .executableTarget(
            name: "Cena",
            path: "Cena",
            exclude: ["Info.plist", "Cena.entitlements"],
            resources: [
                .process("Assets.xcassets")
            ]
        )
    ]
)
