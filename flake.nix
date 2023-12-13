{
  description = "A Nix-flake-based gammaloop development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
  }: let
    overlays = [
      rust-overlay.overlays.default
      (final: prev: {
        rustToolchain = let
          rust = prev.rust-bin;
        in
          if builtins.pathExists ./rust-toolchain.toml
          then rust.fromRustupToolchainFile ./rust-toolchain.toml
          else if builtins.pathExists ./rust-toolchain
          then rust.fromRustupToolchainFile ./rust-toolchain
          else rust.stable.latest.default;
      })
    ];
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f {
          pkgs = import nixpkgs {inherit overlays system;};
        });
  in {
    devShells = forEachSupportedSystem ({pkgs}: {
      default = pkgs.mkShell {
        #devshell definition :
        # LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";

        packages = with pkgs; [
          rustToolchain
          openssl
          gnum4
          gmp.dev
          mpfr.dev
          gcc_debug.out
          stdenv.cc.cc.lib
          pkg-config
          cargo-deny
          cargo-edit
          cargo-watch
          python311
          texlive.combined.scheme-medium
          poetry
          rust-analyzer
          maturin
        ];
      };
    });
  };
}
