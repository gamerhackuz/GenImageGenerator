"""
🎨 PyTorch GAN Image Generator
================================
Rasm beriladi → Label yoziladi → O'rgatiladi → Yangi rasm generatsiya qilinadi
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox, Slider
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Devic
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 50
LATENT_DIM = 64
CHANNELS = 3

# ─────────────────────────────────────────────
# Genaratr
# ─────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim, channels, img_size):
        super().__init__()
        self.init_size = img_size // 4  # 12
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.model(out)
        # Resize to exact IMG_SIZE
        img = torch.nn.functional.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return img

# ─────────────────────────────────────────────
# Diskiymator
# ─────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, channels, img_size):
        super().__init__()

        def block(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(channels, 64, bn=False),
            *block(64, 128),
            *block(128, 256),
            nn.Flatten(),
        )

    
        dummy = torch.zeros(1, channels, img_size, img_size)
        flat_size = self.model(dummy).shape[1]

        self.adv_layer = nn.Sequential(
            nn.Linear(flat_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.model(img)
        validity = self.adv_layer(features)
        return validity

# ─────────────────────────────────────────────
# Dotoset
# ─────────────────────────────────────────────
class LabeledImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img

# ─────────────────────────────────────────────
# main a pp
# ─────────────────────────────────────────────
class GANApp:
    def __init__(self):
        self.images_data = {}      
        self.generators = {}      
        self.discriminators = {}   
        self.training_log = []
        self.current_label = None
        self.epochs_per_train = 50

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self._build_ui()

    # ──────────────────────────────────────
    # UI
    # ──────────────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
        self.fig.canvas.manager.set_window_title("🎨 GAN Image Generator")

        # taytl
        self.fig.text(0.5, 0.97, "🎨 GAN Image Generator | PyTorch + Matplotlib",
                      ha='center', va='top', fontsize=15, color='white',
                      fontweight='bold', fontfamily='monospace')

        # ── Left panel
        self.ax_input = self.fig.add_axes([0.02, 0.55, 0.25, 0.35])
        self.ax_input.set_facecolor("#1a1a2e")
        self.ax_input.set_title("📂 Yuklangan Rasm", color='#00d4ff', fontsize=10, pad=8)
        self.ax_input.axis('off')
        self._show_placeholder(self.ax_input, "Rasm yuklanmagan")

        # ── Midl
        self.ax_gen = self.fig.add_axes([0.38, 0.55, 0.25, 0.35])
        self.ax_gen.set_facecolor("#1a1a2e")
        self.ax_gen.set_title("✨ Generatsiya qilingan rasm", color='#ff6b9d', fontsize=10, pad=8)
        self.ax_gen.axis('off')
        self._show_placeholder(self.ax_gen, "Hali generatsiya qilinmagan")

        # ── Right
        self.ax_loss = self.fig.add_axes([0.68, 0.55, 0.30, 0.35])
        self.ax_loss.set_facecolor("#0d0f1a")
        self.ax_loss.set_title("📈 Training Loss", color='#ffe066', fontsize=10, pad=8)
        self.ax_loss.tick_params(colors='gray', labelsize=8)
        for spine in self.ax_loss.spines.values():
            spine.set_color('#333')
        self.ax_loss.set_facecolor("#0d0f1a")
        self.g_losses = []
        self.d_losses = []

        
        self.ax_log = self.fig.add_axes([0.02, 0.10, 0.96, 0.38])
        self.ax_log.set_facecolor("#111")
        self.ax_log.axis('off')
        self.ax_log.set_title("📋 Logllar", color='#aaa', fontsize=9, pad=6, loc='left')
        self.log_text = self.ax_log.text(0.01, 0.95, "", transform=self.ax_log.transAxes,
                                          va='top', ha='left', fontsize=7.5, color='#00ff88',
                                          fontfamily='monospace', wrap=True)

        # ── Buton ──
       
        ax_load = self.fig.add_axes([0.02, 0.02, 0.14, 0.05])
        self.btn_load = Button(ax_load, "📂 Rasm Yuklash", color='#1e3a5f', hovercolor='#2a5298')
        self.btn_load.label.set_color('white')
        self.btn_load.label.set_fontsize(9)
        self.btn_load.on_clicked(self._on_load)

        
        ax_label_box = self.fig.add_axes([0.18, 0.02, 0.12, 0.05])
        self.txt_label = TextBox(ax_label_box, 'Label: ', initial='car',
                                  color='#1a1a2e', hovercolor='#2a2a4e')
        self.txt_label.label.set_color('white')
        self.txt_label.text_disp.set_color('#00d4ff')

        
        ax_epochs = self.fig.add_axes([0.32, 0.025, 0.12, 0.035])
        self.slider_epochs = Slider(ax_epochs, 'Epochs ', 10, 500,
                                     valinit=50, valstep=10, color='#ff6b9d')
        self.slider_epochs.label.set_color('white')
        self.slider_epochs.valtext.set_color('#ff6b9d')

       
        ax_train = self.fig.add_axes([0.46, 0.02, 0.14, 0.05])
        self.btn_train = Button(ax_train, "🧠 O'rgatish", color='#1a3d1a', hovercolor='#2d7a2d')
        self.btn_train.label.set_color('#00ff88')
        self.btn_train.label.set_fontsize(9)
        self.btn_train.on_clicked(self._on_train)

        
        ax_gen = self.fig.add_axes([0.62, 0.02, 0.14, 0.05])
        self.btn_gen = Button(ax_gen, "✨ Generate", color='#3d1a3d', hovercolor='#7a2d7a')
        self.btn_gen.label.set_color('#ff6b9d')
        self.btn_gen.label.set_fontsize(9)
        self.btn_gen.on_clicked(self._on_generate)

        
        ax_save = self.fig.add_axes([0.78, 0.02, 0.10, 0.05])
        self.btn_save = Button(ax_save, "💾 Saqlash", color='#3d2a00', hovercolor='#7a5500')
        self.btn_save.label.set_color('#ffe066')
        self.btn_save.label.set_fontsize(9)
        self.btn_save.on_clicked(self._on_save)

        
        self.status_text = self.fig.text(0.5, 0.005, "✅ Tayyor | Rasm yuklang va label kiriting",
                                          ha='center', fontsize=8, color='#888', fontfamily='monospace')

        
        info = ("💡 Qo'llanma:\n"
                "1️⃣  'Rasm Yuklash' → rasm tanlang\n"
                "2️⃣  Label kiriting (masalan: car)\n"
                "3️⃣  Epochs sonini belgilang\n"
                "4️⃣  'O'rgatish' bosing → GAN o'rgatiladi\n"
                "5️⃣  'Generate' bosing → yangi rasm yaratiladi")
        self.fig.text(0.02, 0.50, info, fontsize=7.5, color='#888',
                      fontfamily='monospace', va='top',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#111', edgecolor='#333'))

        plt.show()

   
    def _show_placeholder(self, ax, msg):
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.axis('off')
        ax.text(0.5, 0.5, msg, ha='center', va='center',
                color='#555', fontsize=9, fontfamily='monospace',
                transform=ax.transAxes)
        self.fig.canvas.draw_idle()

    def _log(self, msg):
        self.training_log.append(msg)
        
        lines = self.training_log[-20:]
        self.log_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _set_status(self, msg, color='#888'):
        self.status_text.set_text(msg)
        self.status_text.set_color(color)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _update_loss_chart(self):
        self.ax_loss.clear()
        self.ax_loss.set_facecolor("#0d0f1a")
        self.ax_loss.set_title("📈 Training Loss", color='#ffe066', fontsize=10, pad=8)
        self.ax_loss.tick_params(colors='gray', labelsize=8)
        for spine in self.ax_loss.spines.values():
            spine.set_color('#333')

        if self.g_losses:
            x = range(len(self.g_losses))
            self.ax_loss.plot(x, self.g_losses, color='#ff6b9d', linewidth=1.5, label='Generator')
            self.ax_loss.plot(x, self.d_losses, color='#00d4ff', linewidth=1.5, label='Discriminator')
            self.ax_loss.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
            self.ax_loss.set_xlabel('Epoch', color='gray', fontsize=8)
            self.ax_loss.set_ylabel('Loss', color='gray', fontsize=8)

        self.fig.canvas.draw_idle()

    
    def _on_load(self, event):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Rasm tanlang",
                filetypes=[("Rasm fayllari", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Barchasi", "*.*")]
            )
            root.destroy()

            if not path:
                self._log("⚠️  Rasm tanlanmadi.")
                return

            img = Image.open(path).convert("RGB")
            filename = os.path.basename(path)
            label = self.txt_label.text.strip() or "unknown"

            
            if label not in self.images_data:
                self.images_data[label] = []
            self.images_data[label].append(img)
            self.current_label = label

           
            self.ax_input.clear()
            self.ax_input.set_facecolor("#1a1a2e")
            self.ax_input.axis('off')
            self.ax_input.imshow(np.array(img))
            self.ax_input.set_title(f"📂 {filename}\nLabel: '{label}'",
                                     color='#00d4ff', fontsize=9, pad=6)
            self.fig.canvas.draw_idle()

            count = len(self.images_data[label])
            self._log(f"✅ Rasm yuklandi: {filename}")
            self._log(f"   Label: '{label}' | Jami rasm: {count} ta")
            self._set_status(f"✅ '{label}' label bilan {filename} yuklandi", '#00d4ff')

        except Exception as e:
            self._log(f"❌ Xato: {e}")

    # ──────────────────────────────────────
    # Trayn
    # ──────────────────────────────────────
    def _on_train(self, event):
        label = self.txt_label.text.strip() or "unknown"
        self.current_label = label

        if label not in self.images_data or len(self.images_data[label]) == 0:
            self._log(f"❌ '{label}' uchun rasm yuklanmagan!")
            self._set_status("❌ Avval rasm yuklang!", '#ff4444')
            return

        epochs = int(self.slider_epochs.val)
        self._log(f"\n{'='*50}")
        self._log(f"🧠 '{label}' uchun o'rgatish boshlandi")
        self._log(f"   Epochs: {epochs} | Device: {DEVICE}")
        self._log(f"   Rasmlar soni: {len(self.images_data[label])}")
        self._set_status(f"⏳ O'rgatilmoqda... (0/{epochs})", '#ffe066')

        
        imgs = self.images_data[label]
        repeat = max(1, 100 // len(imgs))
        all_imgs = imgs * repeat

        dataset = LabeledImageDataset(all_imgs, transform=self.transform)
        loader = DataLoader(dataset, batch_size=min(8, len(all_imgs)), shuffle=True, drop_last=False)

        
        G = Generator(LATENT_DIM, CHANNELS, IMG_SIZE).to(DEVICE)
        D = Discriminator(CHANNELS, IMG_SIZE).to(DEVICE)

       
        if label in self.generators:
            G.load_state_dict(self.generators[label].state_dict())
            D.load_state_dict(self.discriminators[label].state_dict())
            self._log("   ♻️  Avvalgi weights yuklandi (continue training)")

        opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        self.g_losses = []
        self.d_losses = []

        for epoch in range(1, epochs + 1):
            g_loss_total = 0.0
            d_loss_total = 0.0
            batches = 0

            for real_imgs in loader:
                real_imgs = real_imgs.to(DEVICE)
                b = real_imgs.size(0)

                real_labels = torch.ones(b, 1).to(DEVICE) * 0.9   
                fake_labels = torch.zeros(b, 1).to(DEVICE)

                
                z = torch.randn(b, LATENT_DIM).to(DEVICE)
                fake_imgs = G(z).detach()

                D.zero_grad()
                loss_real = criterion(D(real_imgs), real_labels)
                loss_fake = criterion(D(fake_imgs), fake_labels)
                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                opt_D.step()

               
                G.zero_grad()
                z = torch.randn(b, LATENT_DIM).to(DEVICE)
                gen_imgs = G(z)
                g_loss = criterion(D(gen_imgs), torch.ones(b, 1).to(DEVICE))
                g_loss.backward()
                opt_G.step()

                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()
                batches += 1

            avg_g = g_loss_total / batches
            avg_d = d_loss_total / batches
            self.g_losses.append(avg_g)
            self.d_losses.append(avg_d)

            
            if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
                bar_done = int(20 * epoch / epochs)
                bar = "█" * bar_done + "░" * (20 - bar_done)
                pct = epoch / epochs * 100
                self._log(f"  [{bar}] {epoch:>4}/{epochs} ({pct:5.1f}%) | G:{avg_g:.4f} D:{avg_d:.4f}")
                self._set_status(f"⏳ O'rgatilmoqda: {epoch}/{epochs} epochs ({pct:.0f}%)", '#ffe066')
                self._update_loss_chart()

        
        self.generators[label] = G
        self.discriminators[label] = D

        self._log(f"\n✅ O'rgatish tugadi! '{label}' generator tayyor.")
        self._log(f"   Jami {epochs} epoch, {epochs * batches} batch")
        self._set_status(f"✅ '{label}' o'rgatildi! Endi 'Generate' bosing.", '#00ff88')
        self._update_loss_chart()

    # ──────────────────────────────────────
    # Generat
    # ──────────────────────────────────────
    def _on_generate(self, event):
        label = self.txt_label.text.strip() or self.current_label

        if label not in self.generators:
            self._log(f"❌ '{label}' hali o'rgatilmagan! Avval 'O'rgatish' bosing.")
            self._set_status("❌ Avval o'rgatish kerak!", '#ff4444')
            return

        G = self.generators[label]
        G.eval()

        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            gen = G(z).cpu().squeeze(0)  # [3, 50, 50]
            # Denormalize: [-1,1] → [0,1]
            gen = (gen * 0.5 + 0.5).clamp(0, 1)
            gen_np = gen.permute(1, 2, 0).numpy()  # [50, 50, 3]

        self.last_generated = gen_np
        self.last_label = label

        self.ax_gen.clear()
        self.ax_gen.set_facecolor("#1a1a2e")
        self.ax_gen.axis('off')
        self.ax_gen.imshow(gen_np)
        self.ax_gen.set_title(f"✨ Generated: '{label}'\n50×50px | PyTorch GAN",
                               color='#ff6b9d', fontsize=9, pad=6)
        self.fig.canvas.draw_idle()

        self._log(f"\n✨ '{label}' uchun yangi rasm generatsiya qilindi!")
        self._set_status(f"✨ '{label}' rasmi generatsiya qilindi! 💾 Saqlash uchun 'Saqlash' bosing.", '#ff6b9d')
        G.train()

    # ──────────────────────────────────────
    # savee
    # ──────────────────────────────────────
    def _on_save(self, event):
        if not hasattr(self, 'last_generated'):
            self._log("❌ Avval rasm generatsiya qiling!")
            return

        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', True)

            default_name = f"generated_{self.last_label}.png"
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
            )
            root.destroy()

            if path:
                img_uint8 = (self.last_generated * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                pil_img.save(path)
                self._log(f"💾 Saqlandi: {path}")
                self._set_status(f"💾 Saqlandi: {os.path.basename(path)}", '#ffe066')

        except Exception as e:
            self._log(f"❌ Saqlashda xato: {e}")

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("━" * 60)
    print("  🎨 GAN Image Generator | PyTorch + Matplotlib")
    print("━" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Rasm o'lchami: {IMG_SIZE}×{IMG_SIZE}px")
    print(f"  Latent dim: {LATENT_DIM}")
    print("━" * 60)

    # Check dependencies
    deps_ok = True
    for pkg in ["torch", "torchvision", "matplotlib", "PIL", "numpy"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — o'rnatilmagan!")
            deps_ok = False

    if not deps_ok:
        print("\n📦 O'rnatish uchun:")
        print("  pip install torch torchvision matplotlib pillow numpy")
        sys.exit(1)

    print("━" * 60)
    print("  UI ochilmoqda...")
    app = GANApp()
